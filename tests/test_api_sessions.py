"""Step 7 tests (API_LAYER_PLAN.md §7): sessions over HTTP + final integration.

Mirrors tests/test_session_cli.py (the CLI's session wiring matrix) at the
HTTP layer: history loading, continuation semantics (no re-injection of
system/schema, noted via `warnings`), failed-turn persistence rules,
save-failure warnings, and the session_root confinement re-pinned
end-to-end. Provider seam patched at `src.api.app.deps.resolve_provider`,
log seams at `app`/`error_handlers`; the streaming success seam is patched
in the two streaming tests. No network, no API keys.
"""

import json

import pytest
from fastapi.testclient import TestClient

from src.api import app as app_module
from src.api import error_handlers, streaming
from src.api.app import create_app
from src.core.session import load_session_messages
from src.core.types import ToolCall
from src.providers.base import BaseProvider, ChatResponse as ProviderChatResponse


# ---------------------------------------------------------------------------
# Fakes + fixtures (same seam discipline as the Step 5/6 test files)
# ---------------------------------------------------------------------------


class _FakeProvider(BaseProvider):
    """Scripted non-streaming provider: queued responses in order."""

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
        self.calls.append(list(messages))
        return self._scripted.pop(0)

    def chat_stream(self, messages, *, temperature=0.7, max_tokens=512):
        # Used by the two streaming persistence tests via monkeypatching
        # chat_stream on the instance.
        yield from ()


@pytest.fixture()
def fake_provider():
    return _FakeProvider()


def _build_client(tmp_path, fake_provider, monkeypatch, records):
    def _get_fake(name, model=None, timeout=None):
        fake_provider.model = model or "fake-mini"
        fake_provider.timeout = timeout
        return fake_provider

    monkeypatch.setattr(app_module.deps, "resolve_provider", _get_fake)
    monkeypatch.setattr(app_module, "log_request", lambda **kwargs: records.append(("app", kwargs)))
    monkeypatch.setattr(streaming, "log_request", lambda **kwargs: records.append(("streaming", kwargs)))
    monkeypatch.setattr(error_handlers, "log_request", lambda **kwargs: records.append(("errors", kwargs)))
    return TestClient(create_app(session_root=str(tmp_path)))


# ---------------------------------------------------------------------------
# First turn / second turn (mirrors test_session_cli's first two tests)
# ---------------------------------------------------------------------------


class TestSessionCreationAndContinuation:
    def test_first_turn_creates_the_file_with_the_full_transcript(self, tmp_path, fake_provider, monkeypatch):
        records = []
        client = _build_client(tmp_path, fake_provider, monkeypatch, records)
        fake_provider.enqueue(ProviderChatResponse(text="hello", tokens_out=2, tokens_in=5))

        resp = client.post("/v1/chat", json={"provider": "fake", "prompt": "hi", "session_path": "s.jsonl"})
        assert resp.status_code == 200
        assert resp.json()["warnings"] == []

        saved = load_session_messages(str(tmp_path / "s.jsonl"))
        assert [(m.role, m.content) for m in saved] == [("user", "hi"), ("assistant", "hello")]

    def test_second_turn_continues_the_conversation(self, tmp_path, fake_provider, monkeypatch):
        records = []
        client = _build_client(tmp_path, fake_provider, monkeypatch, records)
        fake_provider.enqueue(
            ProviderChatResponse(text="first reply", tokens_out=2),
            ProviderChatResponse(text="second reply", tokens_out=2),
        )

        assert client.post("/v1/chat", json={"provider": "fake", "prompt": "hi", "session_path": "s.jsonl"}).status_code == 200
        resp = client.post("/v1/chat", json={"provider": "fake", "prompt": "and more", "session_path": "s.jsonl"})
        assert resp.status_code == 200

        # The second request saw the first turn's transcript, not just its own prompt.
        assert [[m.content for m in call] for call in fake_provider.calls] == [
            ["hi"],
            ["hi", "first reply", "and more"],
        ]
        assert [m.content for m in load_session_messages(str(tmp_path / "s.jsonl"))] == [
            "hi",
            "first reply",
            "and more",
            "second reply",
        ]
        # No continuation notes: neither request sent system/schema.
        assert all("warnings" not in kwargs for _, kwargs in records)

    def test_repeating_the_same_prompt_appends_turn_two(self, tmp_path, fake_provider, monkeypatch):
        # Not an idempotent re-save: prompt-form + session_path means a NEW
        # turn each call (the transcript grows by [user, assistant] per call),
        # which is exactly what the CLI does. Prefix-match append only makes
        # a save a no-op when the would-be transcript matches what's stored.
        records = []
        client = _build_client(tmp_path, fake_provider, monkeypatch, records)
        fake_provider.enqueue(
            ProviderChatResponse(text="hello", tokens_out=1),
            ProviderChatResponse(text="hello again", tokens_out=1),
        )
        body = {"provider": "fake", "prompt": "hi", "session_path": "s.jsonl"}
        client.post("/v1/chat", json=body)
        client.post("/v1/chat", json=body)
        assert [m.content for m in load_session_messages(str(tmp_path / "s.jsonl"))] == [
            "hi",
            "hello",
            "hi",
            "hello again",
        ]

    def test_messages_form_can_open_a_session_too(self, tmp_path, fake_provider, monkeypatch):
        records = []
        client = _build_client(tmp_path, fake_provider, monkeypatch, records)
        fake_provider.enqueue(ProviderChatResponse(text="noted", tokens_out=1))
        resp = client.post(
            "/v1/chat",
            json={
                "provider": "fake",
                "messages": [{"role": "user", "content": "call me Bob"}],
                "session_path": "s.jsonl",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["warnings"] == []
        assert [m.content for m in load_session_messages(str(tmp_path / "s.jsonl"))] == ["call me Bob", "noted"]


# ---------------------------------------------------------------------------
# Continuation semantics: system / schema not re-injected, noted via warnings
# ---------------------------------------------------------------------------


class TestContinuationSemantics:
    def test_system_is_not_re_injected_and_the_ignition_is_warned(self, tmp_path, fake_provider, monkeypatch):
        records = []
        client = _build_client(tmp_path, fake_provider, monkeypatch, records)
        fake_provider.enqueue(
            ProviderChatResponse(text="first", tokens_out=1),
            ProviderChatResponse(text="second", tokens_out=1),
        )

        client.post(
            "/v1/chat",
            json={"provider": "fake", "system": "You are terse.", "prompt": "hi", "session_path": "s.jsonl"},
        )
        resp = client.post(
            "/v1/chat",
            json={"provider": "fake", "system": "You are loud.", "prompt": "again", "session_path": "s.jsonl"},
        )
        assert resp.status_code == 200

        # The CLI's exact rule: the saved system message takes precedence.
        second_call = fake_provider.calls[1]
        assert [m.role for m in second_call] == ["system", "user", "assistant", "user"]
        assert second_call[0].content == "You are terse."

        # Over HTTP the stderr note becomes a warning on the response body.
        assert resp.json()["warnings"] == [
            "`system` is ignored when continuing an existing session; "
            "the session's saved system message takes precedence."
        ]

    def test_first_turn_with_system_warns_nothing(self, tmp_path, fake_provider, monkeypatch):
        # The CLI's exact condition: the note fires only when prior history
        # EXISTS (a continuation). On a first turn --system is honored, so
        # no note — the session file doesn't exist yet.
        records = []
        client = _build_client(tmp_path, fake_provider, monkeypatch, records)
        fake_provider.enqueue(ProviderChatResponse(text="ok", tokens_out=1))
        resp = client.post(
            "/v1/chat",
            json={"provider": "fake", "system": "You are terse.", "prompt": "hi", "session_path": "s.jsonl"},
        )
        assert resp.status_code == 200
        assert resp.json()["warnings"] == []
        assert [m.role for m in fake_provider.calls[0]] == ["system", "user"]

    def test_schema_is_not_re_injected_on_continuation_and_is_warned(self, tmp_path, fake_provider, monkeypatch):
        records = []
        client = _build_client(tmp_path, fake_provider, monkeypatch, records)
        schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        fake_provider.enqueue(
            ProviderChatResponse(text='{"name": "Bob"}', tokens_out=4),
            ProviderChatResponse(text='{"name": "Bob"}', tokens_out=4),
        )

        client.post("/v1/chat", json={"provider": "fake", "prompt": "hi", "schema": schema, "session_path": "s.jsonl"})
        resp = client.post("/v1/chat", json={"provider": "fake", "prompt": "again", "schema": schema, "session_path": "s.jsonl"})
        assert resp.status_code == 200

        # The CLI's exact rule: the schema instruction appears exactly once.
        second_call = fake_provider.calls[1]
        system_contents = [m.content for m in second_call if m.role == "system"]
        assert len(system_contents) == 1
        assert resp.json()["warnings"] == [
            "`schema` is not re-injected when continuing an existing session; "
            "the session's saved schema instruction takes precedence."
        ]
        # The coercion still applied on the continuation (runtime behavior,
        # not re-tested deeply here — the shape rides on the transcript).
        assert resp.json()["data"] == {"name": "Bob"}


# ---------------------------------------------------------------------------
# Failure semantics (contract #5)
# ---------------------------------------------------------------------------


class TestFailedTurnPersistence:
    def test_failed_turn_leaves_the_file_untouched_and_a_retry_resumes(
        self, tmp_path, fake_provider, monkeypatch
    ):
        records = []
        client = _build_client(tmp_path, fake_provider, monkeypatch, records)
        fake_provider.enqueue(
            ProviderChatResponse(text="ok", tokens_out=1),
            ProviderChatResponse(text="recovered", tokens_out=1),
        )

        assert client.post("/v1/chat", json={"provider": "fake", "prompt": "one", "session_path": "s.jsonl"}).status_code == 200

        # Turn 2 fails at the provider (extraction-style failure → 422).
        def _boom(messages, **kwargs):
            from src.core.errors import ExtractionError

            raise ExtractionError("model never satisfied the schema", provider="fake")

        fake_provider.chat = _boom
        resp = client.post("/v1/chat", json={"provider": "fake", "prompt": "two", "session_path": "s.jsonl"})
        assert resp.status_code == 422
        before = (tmp_path / "s.jsonl").read_text(encoding="utf-8")

        # Retry resumes cleanly from turn 1's state (fresh scripted answer).
        fake_provider.chat = _FakeProvider.chat.__get__(fake_provider, _FakeProvider)
        fake_provider.enqueue(ProviderChatResponse(text="recovered2", tokens_out=1))
        assert client.post("/v1/chat", json={"provider": "fake", "prompt": "two", "session_path": "s.jsonl"}).status_code == 200

        contents = [m.content for m in load_session_messages(str(tmp_path / "s.jsonl"))]
        assert contents == ["one", "ok", "two", "recovered2"]


# ---------------------------------------------------------------------------
# Save-failure warnings (the CLI's warn-not-fail rule over HTTP)
# ---------------------------------------------------------------------------


class TestSaveFailureWarnings:
    def test_read_only_dir_save_failure_returns_200_with_warnings(
        self, tmp_path, fake_provider, monkeypatch
    ):
        records = []
        client = _build_client(tmp_path, fake_provider, monkeypatch, records)
        fake_provider.enqueue(ProviderChatResponse(text="hello", tokens_out=2))
        session_file = tmp_path / "s.jsonl"
        session_file.write_text('{"role": "user", "content": "prior"}\n', encoding="utf-8")

        import os

        with monkeypatch.context() as m:
            m.setattr(os, "chmod", os.chmod, raising=False)  # placeholder no-op to keep context API shape
            m.setattr(
                "src.core.session.SessionStore.append",
                lambda self, messages: (_ for _ in ()).throw(OSError("disk full")),
            )
            resp = client.post("/v1/chat", json={"provider": "fake", "prompt": "hi", "session_path": "s.jsonl"})

        assert resp.status_code == 200  # the answer itself succeeded
        body = resp.json()
        assert body["text"] == "hello"
        assert body["warnings"] == ["Could not update session file " + str(session_file) + ": disk full"]
        # The turn otherwise behaved like a success (one success record).
        assert [kwargs for seam, kwargs in records if seam == "app" and kwargs["status"] == "success"]

    def test_streaming_save_failure_populates_the_done_events_warnings(
        self, tmp_path, fake_provider, monkeypatch
    ):
        records = []
        client = _build_client(tmp_path, fake_provider, monkeypatch, records)

        class _Chunked(BaseProvider):
            name = "fake"

            def __init__(self, model="fake-mini", timeout=None):
                self.model = model

            def chat(self, messages, **kwargs):
                raise NotImplementedError

            def chat_stream(self, messages, *, temperature=0.7, max_tokens=512):
                yield "Hi "
                yield "there"

        monkeypatch.setattr(
            app_module.deps, "resolve_provider", lambda name, model=None, timeout=None: _Chunked()
        )
        (tmp_path / "s.jsonl").write_text('{"role": "user", "content": "prior"}\n', encoding="utf-8")

        import os

        with monkeypatch.context() as m:
            m.setattr(
                "src.core.session.SessionStore.append",
                lambda self, messages: (_ for _ in ()).throw(OSError("disk full")),
            )
            with client.stream(
                "POST", "/v1/chat/stream", json={"provider": "fake", "prompt": "hi", "session_path": "s.jsonl"}
            ) as resp:
                assert resp.status_code == 200
                raw = b"".join(resp.iter_raw()).decode("utf-8")

        frames = [f for f in raw.split("\n\n") if f.strip()]
        done = json.loads(frames[-2][len("data: "):])
        assert done["done"] is True
        assert done["text"] == "Hi there"
        assert done["warnings"] == ["Could not update session file " + str(tmp_path / "s.jsonl") + ": disk full"]
        assert frames[-1] == "data: [DONE]"


# ---------------------------------------------------------------------------
# End-to-end confinement + load errors (deps-level pins re-tested through HTTP)
# ---------------------------------------------------------------------------


class TestConfinementEndToEnd:
    def test_session_path_outside_root_is_400_before_any_provider_call(
        self, tmp_path, fake_provider, monkeypatch
    ):
        records = []
        client = _build_client(tmp_path, fake_provider, monkeypatch, records)
        resp = client.post(
            "/v1/chat", json={"provider": "fake", "prompt": "hi", "session_path": "../escape.jsonl"}
        )
        assert resp.status_code == 400
        assert "escapes it" in resp.json()["error"]["message"]
        assert fake_provider.calls == []

    def test_absolute_session_path_is_400(self, tmp_path, fake_provider, monkeypatch):
        records = []
        client = _build_client(tmp_path, fake_provider, monkeypatch, records)
        resp = client.post(
            "/v1/chat",
            json={"provider": "fake", "prompt": "hi", "session_path": str(tmp_path / "abs.jsonl")},
        )
        assert resp.status_code == 400
        assert "absolute path" in resp.json()["error"]["message"]

    def test_sneaky_traversal_is_400(self, tmp_path, fake_provider, monkeypatch):
        records = []
        client = _build_client(tmp_path, fake_provider, monkeypatch, records)
        resp = client.post(
            "/v1/chat", json={"provider": "fake", "prompt": "hi", "session_path": "a/b/../../escape.jsonl"}
        )
        assert resp.status_code == 400

    def test_session_path_without_a_root_is_400(self, monkeypatch, fake_provider):
        records = []

        def _get_fake(name, model=None, timeout=None):
            return fake_provider

        monkeypatch.setattr(app_module.deps, "resolve_provider", _get_fake)
        monkeypatch.setattr(app_module, "log_request", lambda **kwargs: records.append(kwargs))
        client = TestClient(create_app())  # no session_root: feature disabled
        resp = client.post("/v1/chat", json={"provider": "fake", "prompt": "hi", "session_path": "s.jsonl"})
        assert resp.status_code == 400
        assert "disabled" in resp.json()["error"]["message"]

    def test_corrupt_session_file_is_400_with_no_provider_call_and_an_error_log(
        self, tmp_path, fake_provider, monkeypatch
    ):
        records = []
        client = _build_client(tmp_path, fake_provider, monkeypatch, records)
        (tmp_path / "junk.jsonl").write_text("this is not a session file\n", encoding="utf-8")
        resp = client.post(
            "/v1/chat", json={"provider": "fake", "prompt": "hi", "session_path": "junk.jsonl"}
        )
        assert resp.status_code == 400  # the Step 7 guard, pre-provider
        assert "Unusable session file" in resp.json()["error"]["message"]
        assert fake_provider.calls == []
        error_records = [kwargs for seam, kwargs in records if seam == "errors"]
        assert len(error_records) == 1
        assert error_records[0]["error_type"] == "format"
        assert error_records[0]["error_subtype"] == "FormatError"

    def test_confinement_on_the_stream_endpoint_too(self, tmp_path, fake_provider, monkeypatch):
        records = []
        client = _build_client(tmp_path, fake_provider, monkeypatch, records)
        resp = client.post(
            "/v1/chat/stream", json={"provider": "fake", "prompt": "hi", "session_path": "../escape.jsonl"}
        )
        assert resp.status_code == 400  # before the StreamingResponse exists


# ---------------------------------------------------------------------------
# Streaming continuation (mirrors test_session_cli's streaming test)
# ---------------------------------------------------------------------------


class TestStreamingContinuation:
    def test_second_streaming_turn_sees_the_first_transcript(self, tmp_path, monkeypatch):
        records = []
        chunks = {"script": ["Hi", " there"]}

        class _Chunked(BaseProvider):
            name = "fake"
            calls = []

            def __init__(self, model="fake-mini", timeout=None):
                self.model = model

            def chat(self, messages, **kwargs):
                raise NotImplementedError

            def chat_stream(self, messages, *, temperature=0.7, max_tokens=512):
                _Chunked.calls.append(list(messages))
                yield from chunks["script"]

        monkeypatch.setattr(
            app_module.deps, "resolve_provider", lambda name, model=None, timeout=None: _Chunked()
        )
        monkeypatch.setattr(app_module, "log_request", lambda **kwargs: {})
        monkeypatch.setattr(streaming, "log_request", lambda **kwargs: {})
        monkeypatch.setattr(error_handlers, "log_request", lambda **kwargs: {})
        client = TestClient(create_app(session_root=str(tmp_path)))

        assert client.post("/v1/chat/stream", json={"provider": "fake", "prompt": "hi", "session_path": "s.jsonl"}).status_code == 200
        chunks["script"] = ["More"]
        assert client.post("/v1/chat/stream", json={"provider": "fake", "prompt": "more", "session_path": "s.jsonl"}).status_code == 200

        # Second stream saw turn 1's transcript (user + assistant) before its prompt.
        assert [[m.content for m in call] for call in _Chunked.calls] == [
            ["hi"],
            ["hi", "Hi there", "more"],
        ]
        saved = load_session_messages(str(tmp_path / "s.jsonl"))
        assert [m.content for m in saved] == ["hi", "Hi there", "more", "More"]

    def test_failed_stream_after_success_leaves_the_file_at_turn_one(
        self, tmp_path, monkeypatch
    ):
        chunks = {"script": ["ok"]}
        fail = {"on": False}

        class _Chunked(BaseProvider):
            name = "fake"

            def __init__(self, model="fake-mini", timeout=None):
                self.model = model

            def chat(self, messages, **kwargs):
                raise NotImplementedError

            def chat_stream(self, messages, *, temperature=0.7, max_tokens=512):
                if fail["on"]:
                    from src.core.errors import RateLimitError

                    raise RateLimitError("slow down", provider="fake")
                yield from chunks["script"]

        monkeypatch.setattr(
            app_module.deps, "resolve_provider", lambda name, model=None, timeout=None: _Chunked()
        )
        monkeypatch.setattr(app_module, "log_request", lambda **kwargs: {})
        monkeypatch.setattr(streaming, "log_request", lambda **kwargs: {})
        monkeypatch.setattr(error_handlers, "log_request", lambda **kwargs: {})
        client = TestClient(create_app(session_root=str(tmp_path)))

        assert client.post("/v1/chat/stream", json={"provider": "fake", "prompt": "one", "session_path": "s.jsonl"}).status_code == 200
        before = (tmp_path / "s.jsonl").read_text(encoding="utf-8")
        fail["on"] = True
        with client.stream(
            "POST", "/v1/chat/stream", json={"provider": "fake", "prompt": "two", "session_path": "s.jsonl"}
        ) as resp:
            assert resp.status_code == 200  # SSE: status already sent; error arrives in-band
            raw = b"".join(resp.iter_raw()).decode("utf-8")
        assert '"error"' in raw  # the in-band error event
        assert (tmp_path / "s.jsonl").read_text(encoding="utf-8") == before
