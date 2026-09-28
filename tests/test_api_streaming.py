"""Step 6 tests (API_LAYER_PLAN.md §7): src/api/streaming.py + /v1/chat/stream.

TestClient with streamed response iteration over `create_app()`. The
provider seam is patched at `src.api.app.deps.resolve_provider` (the API's
single seam, as in the Step 5 tests) and `log_request` at both the app and
error_handlers seams. No network, no API keys; SSE frames are parsed from
the raw byte stream.
"""

import json

import pytest
from fastapi.testclient import TestClient

from src.api import app as app_module
from src.api import error_handlers, streaming
from src.api.app import create_app
from src.core.errors import RateLimitError
from src.providers.base import BaseProvider


# ---------------------------------------------------------------------------
# Fakes + fixtures
# ---------------------------------------------------------------------------


class _ChunkedProvider(BaseProvider):
    """Streaming fake: yields scripted chunks, optionally raising mid-stream."""

    name = "fake"

    def __init__(self, model: str = "fake-mini", timeout=None):
        self.model = model
        self.timeout = timeout
        self.chunks: list = []
        self.error_after: int | None = None  # raise after N chunks (None = never)
        self.error_factory = lambda: RateLimitError("slow down", provider="fake")
        self.calls: list = []

    def chat(self, messages, **kwargs):
        raise NotImplementedError  # streaming endpoint: chat() is never called

    def chat_stream(self, messages, *, temperature=0.7, max_tokens=512):
        self.calls.append({"messages": list(messages), "temperature": temperature, "max_tokens": max_tokens})
        for i, chunk in enumerate(self.chunks):
            if self.error_after is not None and i == self.error_after:
                raise self.error_factory()
            yield chunk
        if self.error_after is not None and self.error_after >= len(self.chunks):
            raise self.error_factory()


@pytest.fixture()
def fake_provider():
    return _ChunkedProvider()


@pytest.fixture()
def client(fake_provider, monkeypatch):
    """App with the provider seam patched to the shared fake and every log
    seam captured — streaming owns the success record, error_handlers the
    mid-stream error records (app_module's own log seam stays patched too,
    so nothing here can write to the default log file)."""
    records = []

    def _get_fake(name, model=None, timeout=None):
        fake_provider.model = model or "fake-mini"
        fake_provider.timeout = timeout
        return fake_provider

    monkeypatch.setattr(app_module.deps, "resolve_provider", _get_fake)
    monkeypatch.setattr(app_module, "log_request", lambda **kwargs: records.append(("app", kwargs)))
    monkeypatch.setattr(streaming, "log_request", lambda **kwargs: records.append(("streaming", kwargs)))
    monkeypatch.setattr(error_handlers, "log_request", lambda **kwargs: records.append(("errors", kwargs)))
    test_client = TestClient(create_app())
    test_client.records = records  # type: ignore[attr-defined]
    return test_client


def _sse_frames(raw: str) -> list:
    """Parse the raw SSE byte stream into (kind, payload) tuples:
    ('event', dict) for data: {...} frames, ('sentinel', None) for [DONE]."""
    frames = []
    for line in raw.split("\n\n"):
        line = line.strip()
        if not line:
            continue
        assert line.startswith("data: "), f"malformed SSE frame: {line!r}"
        body = line[len("data: "):]
        if body == "[DONE]":
            frames.append(("sentinel", None))
        else:
            frames.append(("event", json.loads(body)))
    return frames


def _stream(client, **body):
    if "prompt" not in body and "messages" not in body:
        body["prompt"] = "Hi"  # the one-of validator requires a message form
    with client.stream("POST", "/v1/chat/stream", json={"provider": "fake", **body}) as resp:
        raw = b"".join(resp.iter_raw()).decode("utf-8")
        return resp.status_code, resp.headers, raw


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestStreamingHappyPath:
    def test_three_chunks_in_order_then_done_then_sentinel(self, client, fake_provider):
        fake_provider.chunks = ["Hel", "lo ", "world"]
        status, headers, raw = _stream(client)
        assert status == 200
        frames = _sse_frames(raw)

        kinds = [kind for kind, _ in frames]
        assert kinds == ["event", "event", "event", "event", "sentinel"]
        deltas = [payload for kind, payload in frames[:3]]
        assert [d["delta"] for d in deltas] == ["Hel", "lo ", "world"]

        done = frames[3][1]
        assert done["done"] is True
        assert done["text"] == "Hello world"  # the join of the chunks
        assert done["warnings"] == []
        assert frames[4] == ("sentinel", None)  # [DONE] last

    def test_content_type_is_text_event_stream(self, client, fake_provider):
        fake_provider.chunks = ["x"]
        _, headers, _ = _stream(client)
        assert headers["content-type"].startswith("text/event-stream")
        assert headers["cache-control"] == "no-cache"
        assert headers["x-accel-buffering"] == "no"

    def test_success_log_record_uses_stream_counting_rules(self, client, fake_provider):
        fake_provider.chunks = ["Hel", "lo world"]
        _, _, raw = _stream(client)
        assert _sse_frames(raw)[-1] == ("sentinel", None)  # stream completed

        streaming_records = [kwargs for seam, kwargs in client.records if seam == "streaming"]
        assert len(streaming_records) == 1  # exactly one success record
        record = streaming_records[0]
        assert record["status"] == "success"
        assert record["provider"] == "fake"
        joined = "Hello world"
        # Streaming never gets provider-billed counts (contract): the
        # client-side count of the joined text, method-labeled.
        assert record["tokens_out"] == streaming.count_tokens(joined)
        assert record["tokens_in"] > 0  # the route's pre-count
        assert record["token_count_method"] in ("tiktoken", "heuristic")
        assert record["error_type"] is None

    def test_done_event_usage_matches_the_log_record(self, client, fake_provider):
        fake_provider.chunks = ["a", "b"]
        _, _, raw = _stream(client)
        done = _sse_frames(raw)[2][1]
        record = [kwargs for seam, kwargs in client.records if seam == "streaming"][0]
        assert done["usage"]["tokens_in"] == record["tokens_in"]
        assert done["usage"]["tokens_out"] == record["tokens_out"]
        assert done["usage"]["token_count_method"] == record["token_count_method"]

    def test_provider_receives_the_assembled_messages(self, client, fake_provider):
        fake_provider.chunks = ["ok"]
        _stream(client, system="Be terse.", prompt="Hi")
        sent = fake_provider.calls[0]["messages"]
        assert [(m.role, m.content) for m in sent] == [("system", "Be terse."), ("user", "Hi")]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestStreamingErrors:
    def test_rate_limit_after_one_chunk_yields_error_event_then_closes(self, client, fake_provider):
        fake_provider.chunks = ["partial "]
        fake_provider.error_after = 1
        status, _, raw = _stream(client)

        assert status == 200  # SSE: the status was already sent with the headers
        frames = _sse_frames(raw)
        assert frames[0] == ("event", {"delta": "partial "})  # the real chunk survived
        error_event = frames[1][1]
        assert error_event["error"]["type"] == "rate_limit"
        assert error_event["error"]["subtype"] == "RateLimitError"
        assert "slow down" in error_event["error"]["message"]
        assert frames[2] == ("sentinel", None)  # stream closes cleanly

        error_records = [kwargs for seam, kwargs in client.records if seam == "errors"]
        assert len(error_records) == 1  # exactly one error record (error_handlers seam)
        assert error_records[0]["status"] == "error"
        assert error_records[0]["error_type"] == "rate_limit"
        assert error_records[0]["error_subtype"] == "RateLimitError"
        assert [kwargs for seam, kwargs in client.records if seam == "streaming"] == []  # no success record

    def test_provider_raising_before_any_chunk_yields_error_event_only(self, client, fake_provider):
        fake_provider.chunks = ["never yielded"]
        fake_provider.error_after = 0
        _, _, raw = _stream(client)
        frames = _sse_frames(raw)
        assert len(frames) == 2
        assert frames[0][0] == "event"
        assert "delta" not in frames[0][1]  # no delta events at all
        assert frames[0][1]["error"]["type"] == "rate_limit"
        assert frames[1] == ("sentinel", None)

    def test_unexpected_exception_becomes_unknown_error_event(self, client, fake_provider):
        fake_provider.chunks = ["x"]
        fake_provider.error_after = 1
        fake_provider.error_factory = lambda: RuntimeError("boom")
        _, _, raw = _stream(client)
        error_event = _sse_frames(raw)[1][1]
        assert error_event["error"]["type"] == "unknown"  # to_gateway_error idempotent path
        assert error_event["error"]["message"] == "boom"  # no raw traceback in the frame


# ---------------------------------------------------------------------------
# Route-level rejection (D8 re-pinned end-to-end)
# ---------------------------------------------------------------------------


class TestStreamRouteRejections:
    def test_tools_rejected_422_before_the_response_starts(self, client, fake_provider):
        resp = client.post("/v1/chat/stream", json={"provider": "fake", "prompt": "Hi", "tools": ["calculator"]})
        assert resp.status_code == 422  # FastAPI body validation, not an SSE error event
        assert "detail" in resp.json()
        assert fake_provider.calls == []

    def test_schema_rejected_422_before_the_response_starts(self, client, fake_provider):
        resp = client.post(
            "/v1/chat/stream",
            json={"provider": "fake", "prompt": "Hi", "schema": {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}},
        )
        assert resp.status_code == 422
        assert fake_provider.calls == []

    def test_malformed_body_is_422_with_no_provider_call(self, client, fake_provider):
        resp = client.post("/v1/chat/stream", json={"provider": "fake"})
        assert resp.status_code == 422
        assert fake_provider.calls == []
        assert client.records == []

    def test_unknown_provider_is_400_before_the_response_starts(self, monkeypatch):
        # Unpatched provider resolution: the real registry path must reject
        # before the StreamingResponse (and its 200 + SSE headers) exists.
        monkeypatch.setattr(app_module, "log_request", lambda **kwargs: {})
        monkeypatch.setattr(error_handlers, "log_request", lambda **kwargs: {})
        local_client = TestClient(create_app())
        resp = local_client.post("/v1/chat/stream", json={"provider": "no_such_provider", "prompt": "Hi"})
        assert resp.status_code == 400
        assert "Available providers" in resp.json()["error"]["message"]

    def test_stream_field_pinned_true_rejects_stream_false(self, client):
        # The endpoint IS the stream request; stream=false contradicts it.
        resp = client.post("/v1/chat/stream", json={"provider": "fake", "prompt": "Hi", "stream": False})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Session save after a successful stream (full matrix stays Step 7)
# ---------------------------------------------------------------------------


class TestStreamingSessionSave:
    def test_successful_stream_persists_the_transcript_including_the_assistant_message(self, tmp_path, monkeypatch, fake_provider):
        session_file = tmp_path / "s.jsonl"
        monkeypatch.setattr(app_module.deps, "resolve_provider", lambda name, model=None, timeout=None: fake_provider)
        monkeypatch.setattr(app_module, "log_request", lambda **kwargs: {})
        monkeypatch.setattr(streaming, "log_request", lambda **kwargs: {})
        monkeypatch.setattr(error_handlers, "log_request", lambda **kwargs: {})
        local_client = TestClient(create_app(session_root=str(tmp_path)))

        fake_provider.chunks = ["Hi", " there"]
        with local_client.stream(
            "POST", "/v1/chat/stream", json={"provider": "fake", "prompt": "hello", "session_path": "s.jsonl"}
        ) as resp:
            assert resp.status_code == 200

        lines = session_file.read_text(encoding="utf-8").strip().split("\n")
        saved = [json.loads(line) for line in lines]
        assert [m["role"] for m in saved] == ["user", "assistant"]
        assert saved[1]["content"] == "Hi there"  # the joined stream, not per-chunk lines

    def test_failed_stream_never_persists(self, tmp_path, monkeypatch, fake_provider):
        session_file = tmp_path / "s.jsonl"
        session_file.write_text('{"role": "user", "content": "prior"}\n', encoding="utf-8")
        monkeypatch.setattr(app_module.deps, "resolve_provider", lambda name, model=None, timeout=None: fake_provider)
        monkeypatch.setattr(app_module, "log_request", lambda **kwargs: {})
        monkeypatch.setattr(streaming, "log_request", lambda **kwargs: {})
        monkeypatch.setattr(error_handlers, "log_request", lambda **kwargs: {})
        local_client = TestClient(create_app(session_root=str(tmp_path)))

        fake_provider.chunks = ["partial"]
        fake_provider.error_after = 1
        with local_client.stream(
            "POST", "/v1/chat/stream", json={"provider": "fake", "prompt": "continue", "session_path": "s.jsonl"}
        ) as resp:
            assert resp.status_code == 200

        # Contract #5: a failed turn leaves the session file byte-identical.
        assert session_file.read_text(encoding="utf-8") == '{"role": "user", "content": "prior"}\n'
