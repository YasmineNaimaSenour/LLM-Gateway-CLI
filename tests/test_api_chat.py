"""Step 5 tests (API_LAYER_PLAN.md §7): src/api/app.py routes.

TestClient over ``create_app()`` — the API-layer analogue of the CLI wiring
tests. The provider seam is patched at ``src.api.app.deps.resolve_provider``
(the API's single seam, mirroring how CLI tests patch ``build_provider``)
and ``log_request`` at the ``src.api.app`` seam. No network, no API keys;
runtime semantics stay covered by the deep tests and are not re-tested
through HTTP.
"""

import json

import pytest
from fastapi.testclient import TestClient

from src.api import app as app_module
from src.api.app import create_app
from src.core.errors import ExtractionError
from src.core.types import ToolCall
from src.providers.base import BaseProvider, ChatResponse as ProviderChatResponse
from src.providers.registry import DEFAULT_REGISTRY as PROVIDER_REGISTRY, ProviderSpec
from src.tools.registry import DEFAULT_REGISTRY as TOOL_REGISTRY

VALID_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
}


# ---------------------------------------------------------------------------
# Fakes + fixtures
# ---------------------------------------------------------------------------


class _FakeProvider(BaseProvider):
    """Scripted provider: returns queued responses in order; records calls."""

    name = "fake"

    def __init__(self, model: str = "fake-mini", timeout=None):
        self.model = model
        self.timeout = timeout
        self.calls: list = []
        self._scripted: list = []

    def enqueue(self, *responses: ProviderChatResponse) -> "_FakeProvider":
        self._scripted = list(responses)
        return self

    def chat(self, messages, *, temperature=0.7, max_tokens=512, tools=None, response_schema=None):
        self.calls.append({"messages": list(messages), "temperature": temperature, "max_tokens": max_tokens})
        return self._scripted.pop(0)

    def chat_stream(self, messages, *, temperature=0.7, max_tokens=512):
        yield from ()  # not exercised in Step 5 (streaming is Step 6)


@pytest.fixture()
def provider_registry_snapshot():
    snap = PROVIDER_REGISTRY.snapshot()
    yield PROVIDER_REGISTRY
    PROVIDER_REGISTRY.restore(snap)


@pytest.fixture()
def tool_registry_snapshot():
    snap = TOOL_REGISTRY.snapshot()
    yield TOOL_REGISTRY
    TOOL_REGISTRY.restore(snap)


@pytest.fixture()
def fake_provider(provider_registry_snapshot):
    provider = _FakeProvider()
    PROVIDER_REGISTRY._providers["fake"] = ProviderSpec(name="fake", cls=_FakeProvider, default_model="fake-mini")
    return provider


@pytest.fixture()
def client(fake_provider, monkeypatch):
    """App with the provider seam patched to the shared fake and
    log_request captured at the app seam."""
    records = []

    def _get_fake(name, model=None, timeout=None):
        fake_provider.model = model or "fake-mini"
        fake_provider.timeout = timeout
        return fake_provider

    monkeypatch.setattr(app_module.deps, "resolve_provider", _get_fake)
    monkeypatch.setattr(app_module, "log_request", lambda **kwargs: records.append(kwargs))
    client = TestClient(create_app())
    client.records = records  # type: ignore[attr-defined]
    return client


# ---------------------------------------------------------------------------
# GET /healthz, GET /v1/providers, GET /v1/tools
# ---------------------------------------------------------------------------


class TestHealthz:
    def test_returns_ok_without_touching_provider_or_log(self, client, fake_provider):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        assert fake_provider.calls == []  # no provider call
        assert client.records == []  # no log record


class TestProvidersEndpoint:
    def test_lists_registry_names_with_default_models(self, client, provider_registry_snapshot):
        # Registry-driven, not hardcoded strings: a private registration shows up.
        class _Extra(_FakeProvider):
            name = "extra"

        PROVIDER_REGISTRY._providers["extra"] = ProviderSpec(name="extra", cls=_Extra, default_model="x-1")
        resp = client.get("/v1/providers")
        assert resp.status_code == 200
        entries = {p["name"]: p["default_model"] for p in resp.json()["providers"]}
        assert entries["fake"] == "fake-mini"
        assert entries["extra"] == "x-1"
        assert resp.json()["providers"] == sorted(resp.json()["providers"], key=lambda p: p["name"])


class TestToolsEndpoint:
    def test_matches_the_tool_registry(self, client, tool_registry_snapshot):
        resp = client.get("/v1/tools")
        assert resp.status_code == 200
        by_name = {t["name"]: t for t in resp.json()["tools"]}
        assert set(by_name) == {"calculator", "current_time"}
        assert by_name["calculator"]["parameters"]["type"] == "object"
        assert set(by_name["calculator"]["parameters"]["properties"]) == {"a", "b", "operation"}
        assert by_name["calculator"]["description"]


# ---------------------------------------------------------------------------
# POST /v1/chat — happy paths
# ---------------------------------------------------------------------------


class TestChatPlainPromptForm:
    def test_happy_path_returns_text_transcript_usage_and_one_success_log(self, client, fake_provider):
        fake_provider.enqueue(ProviderChatResponse(text="Hello!", tokens_out=4, tokens_in=11))
        resp = client.post("/v1/chat", json={"provider": "fake", "prompt": "Hi"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["text"] == "Hello!"
        assert body["data"] is None
        assert [m["role"] for m in body["messages"]] == ["user", "assistant"]
        assert body["messages"][-1]["content"] == "Hello!"
        assert body["provider"] == "fake"
        assert body["model"] == "fake-mini"  # the registry-resolved default
        assert body["tool_call_count"] == 0 and body["tool_iterations"] == 0
        assert body["attempts"] == 1
        # Provider-billed tokens_in wins; no method label (D12).
        assert body["usage"] == {"tokens_in": 11, "tokens_out": 4, "token_count_method": None}
        assert body["warnings"] == []

        # Exactly one success record with the CLI-identical fields (D12).
        assert len(client.records) == 1
        record = client.records[0]
        assert record["status"] == "success"
        assert record["provider"] == "fake"
        assert record["tokens_in"] == 11
        assert record["tokens_out"] == 4
        assert record["temperature"] == 0.7
        assert record["token_count_method"] is None
        assert record["error_type"] is None
        assert record["tool_calls"] == 0 and record["tool_iterations"] is None or record["tool_iterations"] == 0

    def test_provider_received_the_built_user_message(self, client, fake_provider):
        fake_provider.enqueue(ProviderChatResponse(text="ok", tokens_out=1))
        client.post("/v1/chat", json={"provider": "fake", "prompt": "Hi"})
        sent = fake_provider.calls[0]["messages"]
        assert [m.role for m in sent] == ["user"]
        assert sent[0].content == "Hi"


class TestChatFullHistoryForm:
    def test_client_messages_are_passed_through_in_order(self, client, fake_provider):
        fake_provider.enqueue(ProviderChatResponse(text="sure", tokens_out=2, tokens_in=9))
        resp = client.post(
            "/v1/chat",
            json={
                "provider": "fake",
                "messages": [
                    {"role": "system", "content": "You are terse."},
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "second"},
                ],
            },
        )
        assert resp.status_code == 200
        sent = fake_provider.calls[0]["messages"]
        assert [(m.role, m.content) for m in sent] == [
            ("system", "You are terse."),
            ("user", "first"),
            ("assistant", "ok"),
            ("user", "second"),
        ]
        # Transcript returned = input + the new assistant turn.
        assert [m["role"] for m in resp.json()["messages"]] == ["system", "user", "assistant", "user", "assistant"]

    def test_messages_form_with_system_field_does_not_inject_a_second_system(self, client, fake_provider):
        # `system` is the convenience form's knob; in the messages form the
        # transcript is authoritative — no extra system message is injected.
        fake_provider.enqueue(ProviderChatResponse(text="ok", tokens_out=1))
        client.post(
            "/v1/chat",
            json={
                "provider": "fake",
                "system": "ignored in messages form",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        sent = fake_provider.calls[0]["messages"]
        assert [m.role for m in sent] == ["user"]


class TestChatSystemPromptForm:
    def test_system_and_prompt_build_system_plus_user(self, client, fake_provider):
        fake_provider.enqueue(ProviderChatResponse(text="ok", tokens_out=1))
        client.post("/v1/chat", json={"provider": "fake", "system": "Be terse.", "prompt": "Hi"})
        sent = fake_provider.calls[0]["messages"]
        assert [(m.role, m.content) for m in sent] == [("system", "Be terse."), ("user", "Hi")]


class TestChatUsageFallback:
    def test_pre_count_fallback_when_provider_omits_usage(self, client, fake_provider):
        fake_provider.enqueue(ProviderChatResponse(text="Hi back", tokens_out=3, tokens_in=None))
        resp = client.post("/v1/chat", json={"provider": "fake", "prompt": "Hi"})
        body = resp.json()
        assert body["usage"]["tokens_in"] > 0  # client-side pre-count, not None/0
        assert body["usage"]["token_count_method"] in ("tiktoken", "heuristic")
        record = client.records[0]
        assert record["tokens_in"] == body["usage"]["tokens_in"]
        assert record["token_count_method"] == body["usage"]["token_count_method"]


# ---------------------------------------------------------------------------
# POST /v1/chat — tools (the loop runs untouched)
# ---------------------------------------------------------------------------


class TestChatWithTools:
    def test_tool_loop_round_trip_executes_the_calculator(self, client, fake_provider):
        # Round 1: model requests a calculator call; Round 2: final answer.
        fake_provider.enqueue(
            ProviderChatResponse(
                text="",
                tokens_out=5,
                tool_calls=[ToolCall(id="call_1", name="calculator", arguments={"a": 3, "b": 9, "operation": "multiply"})],
            ),
            ProviderChatResponse(text="The answer is 27.", tokens_out=6, tokens_in=25),
        )
        resp = client.post("/v1/chat", json={"provider": "fake", "prompt": "3*9?", "tools": ["calculator"]})
        assert resp.status_code == 200
        body = resp.json()

        # Loop stats from run_turn, untouched by the API layer.
        assert body["tool_call_count"] == 1
        assert body["tool_iterations"] == 1
        assert body["text"] == "The answer is 27."
        assert body["usage"]["tokens_in"] == 25  # last call billed on the full conversation

        # The transcript contains the tool message with the executor's real
        # result — proving the resolved executor ran the calculator.
        tool_messages = [m for m in body["messages"] if m["role"] == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["tool_call_id"] == "call_1"
        assert tool_messages[0]["name"] == "calculator"
        assert tool_messages[0]["content"] == "27.0"

        # Provider saw the loop's three rounds: initial call, then the final
        # call carrying the assistant tool_calls + tool result.
        assert len(fake_provider.calls) == 2
        final_call_messages = fake_provider.calls[1]["messages"]
        assert final_call_messages[-1].role == "tool"
        assert final_call_messages[-1].content == "27.0"

        # Success record carries the tool fields, CLI-identical.
        record = client.records[0]
        assert record["tool_calls"] == 1 and record["tool_iterations"] == 1

    def test_unknown_tool_is_400_before_any_provider_call(self, client, fake_provider):
        resp = client.post("/v1/chat", json={"provider": "fake", "prompt": "Hi", "tools": ["no_such_tool"]})
        assert resp.status_code == 400
        assert "Available tools" in resp.json()["error"]["message"]
        assert fake_provider.calls == []  # pre-network
        assert client.records == []  # and unlogged (FormatError raised in a Depends resolver)


# ---------------------------------------------------------------------------
# POST /v1/chat — schema
# ---------------------------------------------------------------------------


class TestChatWithSchema:
    def test_retry_then_valid_json_populates_data_and_attempts(self, client, fake_provider):
        fake_provider.enqueue(
            ProviderChatResponse(text="not json", tokens_out=3, tokens_in=8),
            ProviderChatResponse(text='{"name": "Ada"}', tokens_out=4, tokens_in=15),
        )
        resp = client.post("/v1/chat", json={"provider": "fake", "prompt": "who?", "schema": VALID_SCHEMA})
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"] == {"name": "Ada"}
        assert body["attempts"] == 2
        assert body["text"] == '{"name": "Ada"}'  # the raw final text
        assert body["messages"][-1]["role"] == "assistant"
        assert body["messages"][-1]["content"] == '{"name": "Ada"}'
        assert body["usage"]["tokens_in"] == 15

    def test_inline_schema_validation_errors_map_per_step3(self, client, fake_provider):
        # Broken document → SchemaError → 424 (D11's format-subtype split).
        bad_doc = client.post(
            "/v1/chat",
            json={
                "provider": "fake",
                "prompt": "Hi",
                "schema": {"type": "object", "properties": {"a": {"type": "strin"}}, "required": ["a"]},
            },
        )
        assert bad_doc.status_code == 424
        assert bad_doc.json()["error"]["subtype"] == "SchemaError"

        # Valid JSON Schema but unsupported subset → UnsupportedSchemaError → 422.
        unsupported = client.post(
            "/v1/chat",
            json={
                "provider": "fake",
                "prompt": "Hi",
                "schema": {"type": "object", "properties": {"x": {"oneOf": [{"type": "string"}]}}},
            },
        )
        assert unsupported.status_code == 422
        assert unsupported.json()["error"]["subtype"] == "UnsupportedSchemaError"
        assert "oneOf" in unsupported.json()["error"]["message"]

    def test_schema_errors_happen_before_any_provider_call(self, client, fake_provider):
        client.post("/v1/chat", json={"provider": "fake", "prompt": "Hi", "schema": {"type": "strin"}})
        assert fake_provider.calls == []


# ---------------------------------------------------------------------------
# POST /v1/chat — stream=true and malformed bodies (D8 + built-in validation)
# ---------------------------------------------------------------------------


class TestStreamFlagAndBodyValidation:
    def test_stream_true_on_chat_is_rejected_naming_the_stream_endpoint(self, client, fake_provider):
        # Bare stream=true passes the D8 validator (only stream+tools/schema is
        # invalid there) and hits this route's own guard: a request-shaped
        # FormatError — the D11 map's 400 default — with a message that names
        # the streaming surface. Loud failure, uniform envelope, no provider
        # call, and no silent downgrade to a non-streamed response.
        resp = client.post("/v1/chat", json={"provider": "fake", "prompt": "Hi", "stream": True})
        assert resp.status_code == 400
        assert resp.json()["error"]["subtype"] == "FormatError"
        assert "/v1/chat/stream" in resp.json()["error"]["message"]
        assert fake_provider.calls == []
        assert client.records == []  # resolver-stage rejection: no log record

    def test_stream_plus_tools_rejected_by_the_request_model(self, client, fake_provider):
        resp = client.post(
            "/v1/chat", json={"provider": "fake", "prompt": "Hi", "stream": True, "tools": ["calculator"]}
        )
        assert resp.status_code == 422  # FastAPI's built-in validation error, not the envelope
        assert "detail" in resp.json()
        assert fake_provider.calls == []

    def test_malformed_body_is_fastapi_422_with_no_provider_call_and_no_log(self, client, fake_provider):
        resp = client.post("/v1/chat", json={"provider": "fake"})  # neither messages nor prompt
        assert resp.status_code == 422
        assert "detail" in resp.json()  # FastAPI's built-in shape, kept by the passthrough handler
        assert fake_provider.calls == []
        assert client.records == []

    def test_unknown_provider_is_400_with_registry_message(self, monkeypatch, fake_provider):
        # No fake registered beyond the fixture name: resolve via the real
        # registry path (only log_request is patched) to pin the registry's
        # own message through HTTP.
        records = []
        monkeypatch.setattr(app_module, "log_request", lambda **kwargs: records.append(kwargs))
        client = TestClient(create_app())
        resp = client.post("/v1/chat", json={"provider": "no_such_provider", "prompt": "Hi"})
        assert resp.status_code == 400
        assert "Available providers" in resp.json()["error"]["message"]
        assert records == []  # validation failure in a resolver: no record


# ---------------------------------------------------------------------------
# POST /v1/structured
# ---------------------------------------------------------------------------


class TestStructuredEndpoint:
    def test_happy_path_returns_data_raw_text_and_usage(self, client, fake_provider):
        fake_provider.enqueue(ProviderChatResponse(text='{"person": "Ada Lovelace"}', tokens_out=9, tokens_in=21))
        resp = client.post(
            "/v1/structured",
            json={
                "provider": "fake",
                "text": "Ada wrote the first compiler.",
                "schema": {
                    "type": "object",
                    "properties": {"person": {"type": "string"}},
                    "required": ["person"],
                },
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"] == {"person": "Ada Lovelace"}
        assert body["raw_text"] == '{"person": "Ada Lovelace"}'
        assert body["attempts"] == 1
        assert body["provider"] == "fake" and body["model"] == "fake-mini"
        assert body["usage"]["tokens_in"] == 21

        record = client.records[0]
        assert record["status"] == "success"
        assert record["provider"] == "fake"
        assert record["tokens_in"] == 21
        assert record["tokens_out"] == 9
        assert record["temperature"] == 0.0  # CLI structured default, determinism
        assert "tool_calls" not in record or record["tool_calls"] is None

    def test_provider_receives_extraction_messages_not_raw_text(self, client, fake_provider):
        # extract() builds [system(schema), user(text)]; the API layer passes
        # text+schema to it and lets the runtime shape the messages.
        fake_provider.enqueue(ProviderChatResponse(text='{"person": "Ada"}', tokens_out=3))
        client.post(
            "/v1/structured",
            json={
                "provider": "fake",
                "text": "Ada wrote the first compiler.",
                "schema": {"type": "object", "properties": {"person": {"type": "string"}}, "required": ["person"]},
            },
        )
        sent = fake_provider.calls[0]["messages"]
        assert [m.role for m in sent] == ["system", "user"]
        assert sent[1].content == "Ada wrote the first compiler."
        assert "JSON Schema" in sent[0].content
        assert json.loads(sent[0].content.split("JSON Schema:\n", 1)[1])["type"] == "object"

    def test_extraction_error_maps_to_422(self, client, fake_provider):
        fake_provider.enqueue(
            ProviderChatResponse(text="nope 1", tokens_out=1),
            ProviderChatResponse(text="nope 2", tokens_out=1),
            ProviderChatResponse(text="nope 3", tokens_out=1),
        )
        resp = client.post(
            "/v1/structured",
            json={
                "provider": "fake",
                "text": "irrelevant",
                "schema": {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
            },
        )
        assert resp.status_code == 422
        error = resp.json()["error"]
        assert error["type"] == "format"
        assert error["subtype"] == "ExtractionError"

    def test_invalid_inline_schema_maps_to_424(self, client, fake_provider):
        resp = client.post(
            "/v1/structured",
            json={
                "provider": "fake",
                "text": "irrelevant",
                "schema": {"type": "object", "properties": {"a": {"type": "strin"}}, "required": ["a"]},
            },
        )
        assert resp.status_code == 424
        assert resp.json()["error"]["subtype"] == "SchemaError"
        assert fake_provider.calls == []


# ---------------------------------------------------------------------------
# Sessions: failed turns never persist (contract #5); the full matrix is Step 7
# ---------------------------------------------------------------------------


class TestFailedTurnNeverPersists:
    def test_session_file_untouched_when_the_turn_fails(self, client, fake_provider, tmp_path, monkeypatch):
        session_file = tmp_path / "demo.jsonl"
        session_file.write_text('{"role": "user", "content": "prior"}\n', encoding="utf-8")

        app = create_app(session_root=str(tmp_path))
        records = []

        def _get_fake(name, model=None, timeout=None):
            return fake_provider

        monkeypatch.setattr(app_module.deps, "resolve_provider", _get_fake)
        monkeypatch.setattr(app_module, "log_request", lambda **kwargs: records.append(kwargs))
        local_client = TestClient(app)

        # Continuation turn: provider raises after validation (constructor is
        # fake-safe, so the failure is a scripted chat() blowup).
        def _boom(messages, **kwargs):
            raise ExtractionError("model never satisfied the schema", provider="fake")

        fake_provider.chat = _boom
        fake_provider.enqueue()  # nothing scripted: chat() raises instead

        resp = local_client.post(
            "/v1/chat",
            json={"provider": "fake", "prompt": "continue", "session_path": "demo.jsonl"},
        )
        assert resp.status_code == 422  # ExtractionError via the Step 3 handler
        assert session_file.read_text(encoding="utf-8") == '{"role": "user", "content": "prior"}\n'
        assert records == []  # error record comes from the error_handlers seam, unpatched here → none captured
