"""GatewayError → HTTP: the single choke point for every API error (D6).

The direct analogue of the CLI's `_log_and_report()` (src/cli.py): every
error — expected or not — flows through exactly one of the handlers
registered here, which (a) normalizes any exception via the idempotent
`to_gateway_error()`, (b) appends the same JSONL log record the CLI would
(`status="error"`, `error_type`, `error_subtype` — no ad-hoc logging
anywhere else), and (c) renders the uniform envelope:

    {"error": {"type": <ErrorType>, "subtype": <ClassName>,
               "provider": <name>, "message": <str(exc)>}}

No sixth error category, no per-route try/except for provider failures:
routes raise, this module decides the HTTP status (D11's one-dict rule).
"""

from __future__ import annotations

import requests
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..core.errors import ErrorType, GatewayError, to_gateway_error
from ..core.logger import log_request
from ..core.telemetry import Timer

# ---------------------------------------------------------------------------
# The one mapping (D11): ErrorType → HTTP status.
#
# Debatable rows are deliberately isolated to this dict with per-entry
# comments — changing a mapping later is a one-line diff + one test update.
# `format` is split by subtype at the call site: *request-shaped*
# FormatErrors (unknown provider/tool, broken schema, unusable session) are
# the client's fault → 4xx client-error codes; *model-output-shaped* ones
# (ExtractionError, ToolLoopError) are the provider failing to satisfy the
# request → 422.
# ---------------------------------------------------------------------------

_STATUS_BY_ERROR_TYPE: dict[ErrorType, int] = {
    ErrorType.RATE_LIMIT: 429,  # literal; the retry decision belongs to the client
    ErrorType.CONTEXT_OVERFLOW: 413,  # payload-size problem
    ErrorType.MODEL_ERROR: 502,  # upstream unavailable; 504 is reserved for the read-timeout case below
    ErrorType.FORMAT_ERROR: 400,  # default for request-shaped FormatErrors; overridden per subtype
    ErrorType.UNKNOWN: 500,  # never a raw traceback in the body
}

# Request-shaped FormatError subtypes → specific client-error statuses (D11):
#   424 Failed Dependency — the schema document itself is broken (neither a
#       transport failure nor a normal semantic mismatch; reviewer taste per
#       the plan: a one-line dict entry).
#   422 Unprocessable Entity — the document is valid JSON Schema but uses
#       features outside the supported subset; aligns with FastAPI's own 422
#       for semantically invalid bodies.
_STATUS_BY_FORMAT_SUBTYPE: dict[str, int] = {
    "SchemaError": 424,
    "UnsupportedSchemaError": 422,
    # SessionError and bare FormatError fall through to the ErrorType
    # default (400): wrong server-side reference/name resolution.
}

# Model-output-shaped FormatErrors: the model failed to satisfy the request
# (never satisfied the schema / stuck in a tool loop) → 422 with the subtype
# named in the body (D11).
_MODEL_OUTPUT_FORMAT_SUBTYPES = {"ExtractionError", "ToolLoopError"}


def _status_for(exc: GatewayError) -> int:
    """Derive the HTTP status from the taxonomy — never hand-assigned per route."""
    subtype = type(exc).__name__
    if subtype in _MODEL_OUTPUT_FORMAT_SUBTYPES:
        return 422
    if subtype in _STATUS_BY_FORMAT_SUBTYPE:
        return _STATUS_BY_FORMAT_SUBTYPE[subtype]
    return _STATUS_BY_ERROR_TYPE[exc.error_type]


def _is_read_timeout(exc: GatewayError) -> bool:
    """True when a ModelError was raised from a requests.Timeout cause
    (D11: the upstream *took too long* → 504, not 502)."""
    return isinstance(exc.cause, requests.Timeout)


def build_error_body(exc: GatewayError) -> dict:
    """The uniform envelope. `message` equals str(exc) — the same text the
    CLI would print to stderr (the log/stderr-consistency analogue)."""
    return {
        "error": {
            "type": exc.error_type.value,
            "subtype": type(exc).__name__,
            "provider": exc.provider,
            "message": str(exc),
        }
    }


def log_gateway_error(
    exc: GatewayError,
    *,
    latency_ms: float,
    tokens_in: int = 0,
    temperature: float = 0.0,
    log_path=None,
) -> None:
    """The error-path log record — fields identical to the CLI's
    `_log_and_report()` (same `log_request` schema, status="error").
    `log_path=None` means the logger's own default (env-coupled);
    a path redirects the record for tests without touching that behavior.
    """
    kwargs = {"log_path": log_path} if log_path is not None else {}
    log_request(
        provider=exc.provider or "unknown",
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=0,
        temperature=temperature,
        status="error",
        error_type=exc.error_type.value,
        error_subtype=type(exc).__name__,
        **kwargs,
    )


def register_handlers(app, *, log_path=None) -> None:
    """Register the error handlers on a FastAPI app.

    `log_path` (optional) lets create_app redirect the JSONL records for
    tests without touching LLM_GATEWAY_LOG_PATH semantics (D12).
    """

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(request: Request, exc: GatewayError) -> JSONResponse:
        status = _status_for(exc)
        if status == 502 and _is_read_timeout(exc):
            status = 504
        log_gateway_error(exc, latency_ms=0.0, log_path=log_path)
        return JSONResponse(status_code=status, content=build_error_body(exc))

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        # Idempotent normalization (Step 3 plan): non-Gateway surprises
        # become UnknownError — classified, logged, enveloped; never a raw
        # traceback in the body.
        gateway_exc = to_gateway_error(exc, provider="unknown")
        log_gateway_error(gateway_exc, latency_ms=0.0, log_path=log_path)
        return JSONResponse(status_code=_status_for(gateway_exc), content=build_error_body(gateway_exc))

    # FastAPI's own body-validation 422 and route-not-found 404 are
    # Starlette HTTPExceptions; keep their built-in handling (checked
    # status codes, no GatewayError semantics) rather than enveloping them —
    # the plan's D6 choke point covers GatewayErrors and unexpected
    # exceptions, which are the only things routes raise.
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail}, headers=exc.headers)
