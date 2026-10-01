"""Tests for src/api/error_handlers.py — GatewayError → HTTP status mapping.

TestClient on a tiny probe app whose routes raise — the API-layer analogue
of the CLI error tests. Each probe pins: HTTP status from the D11 map, the
uniform envelope (type/subtype/provider/message), and exactly one
error-path log_request record with the CLI-identical fields. No provider
calls, no network; log_request is patched at the error_handlers seam.
"""

import requests
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import error_handlers
from src.api.error_handlers import build_error_body, register_handlers
from src.core.errors import (
    ContextOverflowError,
    ExtractionError,
    FormatError,
    GatewayError,
    ModelError,
    RateLimitError,
    SchemaError,
    SessionError,
    ToolLoopError,
    UnsupportedSchemaError,
)

# ---------------------------------------------------------------------------
# Probe app: one route per exception scenario
# ---------------------------------------------------------------------------

# path → (exception factory, expected ErrorType value)
PROBES = {
    "/rate-limit": (lambda: RateLimitError("slow down", provider="groq"), "rate_limit"),
    "/context": (lambda: ContextOverflowError("too many tokens", provider="groq"), "context"),
    "/format": (lambda: FormatError("Unknown provider 'nope'. Available providers: groq, ollama", provider=None), "format"),
    "/schema": (lambda: SchemaError("Not a valid JSON Schema: ...", provider=None), "format"),
    "/unsupported-schema": (
        lambda: UnsupportedSchemaError("$.properties.x: uses keyword(s) ['oneOf'] ...", provider=None),
        "format",
    ),
    "/session": (lambda: SessionError("no parseable session records in it", provider=None), "format"),
    "/extraction": (lambda: ExtractionError("Model output did not satisfy the schema after 3 attempt(s)", provider="groq"), "format"),
    "/tool-loop": (lambda: ToolLoopError("Tool loop did not converge after 8 iterations", provider="ollama"), "format"),
    "/model": (lambda: ModelError("Could not reach the API", provider="groq"), "model"),
    "/unexpected": (lambda: RuntimeError("boom"), "unknown"),
}


def _build_probe_app() -> FastAPI:
    app = FastAPI()
    register_handlers(app)
    for path, (factory, _) in PROBES.items():

        def probe(factory=factory):  # sync route (D4); closure binds the factory
            raise factory()

        app.get(path)(probe)

    @app.get("/model-read-timeout")
    def model_read_timeout():
        raise ModelError("Groq request timed out.", provider="groq", cause=requests.exceptions.Timeout())

    @app.get("/model-other-cause")
    def model_other_cause():
        raise ModelError("Could not reach the API.", provider="groq", cause=ValueError("unrelated"))

    return app


@pytest.fixture()
def client():
    # raise_server_exceptions=False: the catch-all `Exception` handler is
    # installed at Starlette's ServerErrorMiddleware level, and TestClient
    # re-raises server exceptions by default — which would mask the handler
    # the production server (uvicorn) uses. This flag exercises it.
    return TestClient(_build_probe_app(), raise_server_exceptions=False)


@pytest.fixture()
def log_records(monkeypatch):
    """Patch log_request at the error_handlers seam; return the captured kwargs."""
    records = []
    monkeypatch.setattr(error_handlers, "log_request", lambda **kwargs: records.append(kwargs))
    return records


# ---------------------------------------------------------------------------
# Status mapping + envelope (D11)
# ---------------------------------------------------------------------------


class TestStatusMapping:
    def test_rate_limit_maps_to_429_with_taxonomy_fields(self, client):
        resp = client.get("/rate-limit")
        assert resp.status_code == 429
        error = resp.json()["error"]
        assert error["type"] == "rate_limit"
        assert error["subtype"] == "RateLimitError"
        assert error["provider"] == "groq"
        assert error["message"] == "slow down"

    def test_context_overflow_maps_to_413(self, client):
        resp = client.get("/context")
        assert resp.status_code == 413
        assert resp.json()["error"]["type"] == "context"

    def test_plain_model_error_maps_to_502(self, client):
        resp = client.get("/model")
        assert resp.status_code == 502
        assert resp.json()["error"]["type"] == "model"

    def test_model_error_with_timeout_cause_maps_to_504(self, client):
        # D11's timeout rule: the upstream took too long → 504, not 502.
        resp = client.get("/model-read-timeout")
        assert resp.status_code == 504
        error = resp.json()["error"]
        assert error["type"] == "model"
        assert error["subtype"] == "ModelError"

    def test_model_error_with_non_timeout_cause_stays_502(self, client):
        resp = client.get("/model-other-cause")
        assert resp.status_code == 502

    @pytest.mark.parametrize(
        "path,status",
        [
            ("/format", 400),  # request-shaped: unknown provider/reference
            ("/session", 400),  # request-shaped: unusable session
            ("/schema", 424),  # the schema document itself is broken
            ("/unsupported-schema", 422),  # valid JSON Schema, outside subset
            ("/extraction", 422),  # model never satisfied the schema
            ("/tool-loop", 422),  # model stuck in a tool loop
        ],
    )
    def test_format_category_splits_by_subtype(self, client, path, status):
        resp = client.get(path)
        assert resp.status_code == status
        assert resp.json()["error"]["type"] == "format"

    def test_unexpected_exception_maps_to_500_with_no_traceback_in_body(self, client):
        resp = client.get("/unexpected")
        assert resp.status_code == 500
        error = resp.json()["error"]
        assert error["type"] == "unknown"
        assert error["message"] == "boom"
        assert "Traceback" not in resp.text

    @pytest.mark.parametrize(
        "path,status",
        [
            ("/rate-limit", 429),
            ("/context", 413),
            ("/model", 502),
            ("/model-read-timeout", 504),
            ("/format", 400),
            ("/schema", 424),
            ("/unsupported-schema", 422),
            ("/extraction", 422),
            ("/tool-loop", 422),
            ("/unexpected", 500),
        ],
    )
    def test_every_probe_resolves_to_its_documented_status(self, client, path, status):
        assert client.get(path).status_code == status


# ---------------------------------------------------------------------------
# Envelope shape + consistency invariant
# ---------------------------------------------------------------------------


class TestEnvelope:
    def test_body_is_exactly_the_uniform_envelope(self, client):
        body = client.get("/rate-limit").json()
        assert set(body) == {"error"}
        assert set(body["error"]) == {"type", "subtype", "provider", "message"}

    @pytest.mark.parametrize(
        "path,expected_message",
        [
            ("/rate-limit", "slow down"),
            ("/extraction", "Model output did not satisfy the schema after 3 attempt(s)"),
            ("/unexpected", "boom"),
        ],
    )
    def test_error_body_message_equals_str_exc(self, client, path, expected_message):
        # The stderr/log-consistency analogue: same text the CLI would print.
        assert client.get(path).json()["error"]["message"] == expected_message

    def test_missing_provider_is_reported_as_unknown(self, client):
        error = client.get("/unexpected").json()["error"]
        assert error["provider"] == "unknown"

    def test_build_error_body_is_pure(self):
        exc = RateLimitError("slow down", provider="groq")
        body = build_error_body(exc)
        assert body == {"error": {"type": "rate_limit", "subtype": "RateLimitError", "provider": "groq", "message": "slow down"}}

    def test_every_gateway_error_still_subclasses_the_base(self):
        # Guard: the probes stay meaningful if the taxonomy grows.
        for factory, _ in PROBES.values():
            exc = factory()
            if isinstance(exc, GatewayError):
                assert hasattr(exc, "error_type")


# ---------------------------------------------------------------------------
# Error-path logging: exactly one record, CLI-identical fields
# ---------------------------------------------------------------------------


class TestErrorPathLogging:
    @pytest.mark.parametrize("path", list(PROBES.keys()))
    def test_every_error_response_is_preceded_by_exactly_one_error_log_record(self, client, log_records, path):
        client.get(path)
        assert len(log_records) == 1
        record = log_records[0]
        assert record["status"] == "error"
        assert record["tokens_in"] == 0
        assert record["tokens_out"] == 0

    @pytest.mark.parametrize(
        "path,expected_type,expected_subtype",
        [
            ("/rate-limit", "rate_limit", "RateLimitError"),
            ("/context", "context", "ContextOverflowError"),
            ("/format", "format", "FormatError"),
            ("/schema", "format", "SchemaError"),
            ("/unsupported-schema", "format", "UnsupportedSchemaError"),
            ("/session", "format", "SessionError"),
            ("/extraction", "format", "ExtractionError"),
            ("/tool-loop", "format", "ToolLoopError"),
            ("/model", "model", "ModelError"),
            ("/unexpected", "unknown", "UnknownError"),
        ],
    )
    def test_log_record_carries_matching_error_type_and_subtype(self, client, log_records, path, expected_type, expected_subtype):
        client.get(path)
        record = log_records[0]
        assert record["error_type"] == expected_type
        assert record["error_subtype"] == expected_subtype

    def test_log_record_provider_matches_the_exception(self, client, log_records):
        client.get("/rate-limit")
        assert log_records[0]["provider"] == "groq"

    def test_unexpected_exception_logs_the_normalized_provider(self, client, log_records):
        client.get("/unexpected")
        assert log_records[0]["provider"] == "unknown"

    def test_log_and_body_come_from_the_same_exception_object(self, client, log_records):
        # The consistency invariant: one exception → both the log fields and
        # the HTTP body, never two divergent derivations.
        client.get("/tool-loop")
        record = log_records[0]
        error = client.get("/tool-loop").json()["error"]
        assert record["error_subtype"] == error["subtype"] == "ToolLoopError"
        assert record["error_type"] == error["type"] == "format"


# ---------------------------------------------------------------------------
# Non-GatewayError HTTP semantics stay FastAPI's own
# ---------------------------------------------------------------------------


class TestHTTPExceptionPassthrough:
    def test_route_not_found_keeps_the_builtin_404_shape(self, client):
        resp = client.get("/definitely-not-a-route")
        assert resp.status_code == 404
        assert resp.json() == {"detail": "Not Found"}

    def test_no_log_record_for_404s(self, client, log_records):
        client.get("/definitely-not-a-route")
        assert log_records == []
