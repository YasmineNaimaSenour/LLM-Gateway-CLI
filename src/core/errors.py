"""Error taxonomy and classification for the LLM Gateway.

Every provider speaks a different error dialect (HTTP status codes, custom
JSON error bodies, connection-level exceptions, ...). This module normalizes
all of that into exactly five categories, which is what the structured
logger and the CLI actually care about:

    RATE_LIMIT | CONTEXT_OVERFLOW | FORMAT_ERROR | MODEL_ERROR | UNKNOWN
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple, Type


class ErrorType(str, Enum): # we use string enums to enable string properties and serialization for the logs
    RATE_LIMIT = "rate_limit"
    CONTEXT_OVERFLOW = "context"
    FORMAT_ERROR = "format"
    MODEL_ERROR = "model"
    UNKNOWN = "unknown"


class GatewayError(Exception):
    """Base class for all normalized gateway errors."""

    error_type: ErrorType = ErrorType.UNKNOWN

    def __init__(self, message: str, *, provider: Optional[str] = None, cause: Optional[BaseException] = None):
        super().__init__(message)
        self.provider = provider
        self.cause = cause


class RateLimitError(GatewayError):
    error_type = ErrorType.RATE_LIMIT


class ContextOverflowError(GatewayError):
    error_type = ErrorType.CONTEXT_OVERFLOW


class FormatError(GatewayError):
    error_type = ErrorType.FORMAT_ERROR


class ModelError(GatewayError):
    error_type = ErrorType.MODEL_ERROR


class UnknownError(GatewayError):
    error_type = ErrorType.UNKNOWN


# ---------------------------------------------------------------------------
# Classification heuristics
#
# Providers rarely agree on wording, so classification uses both HTTP status
# codes (reliable when available) and substring markers in the error message
# (fallback for connection-level or provider-specific errors).
# ---------------------------------------------------------------------------

_RATE_LIMIT_MARKERS: Tuple[str, ...] = ("rate limit", "429", "too many requests", "quota exceeded")
_CONTEXT_MARKERS: Tuple[str, ...] = (
    "context length",
    "maximum context",
    "context_length_exceeded",
    "too many tokens",
    "context window",
    "reduce the length",
)
_MODEL_MARKERS: Tuple[str, ...] = (
    "model not found",
    "model_not_found",
    "does not exist",
    "unknown model",
    "unsupported model",
    "no such model",
)
_FORMAT_MARKERS: Tuple[str, ...] = (
    "invalid json",
    "malformed",
    "decode",
    "validation error",
    "invalid request",
    "invalid api key",
    "unauthorized",
    "missing required",
)


def classify_exception(exc: BaseException, *, status_code: Optional[int] = None) -> ErrorType:
    """Map a raw exception (and optional HTTP status) onto the shared taxonomy."""

    if isinstance(exc, GatewayError): # No need to classify it again. this makes the function idempotent
        return exc.error_type

    message = str(exc).lower()

    if status_code == 429 or _matches(message, _RATE_LIMIT_MARKERS):
        return ErrorType.RATE_LIMIT
    if _matches(message, _CONTEXT_MARKERS):
        return ErrorType.CONTEXT_OVERFLOW
    if status_code == 404 or _matches(message, _MODEL_MARKERS):
        return ErrorType.MODEL_ERROR
    if status_code in (400, 401, 403, 422) or _matches(message, _FORMAT_MARKERS):
        return ErrorType.FORMAT_ERROR

    return ErrorType.UNKNOWN


def _matches(message: str, markers: Tuple[str, ...]) -> bool:
    return any(marker in message for marker in markers)


_ERROR_CLASS_BY_TYPE: dict[ErrorType, Type[GatewayError]] = {
    ErrorType.RATE_LIMIT: RateLimitError,
    ErrorType.CONTEXT_OVERFLOW: ContextOverflowError,
    ErrorType.FORMAT_ERROR: FormatError,
    ErrorType.MODEL_ERROR: ModelError,
    ErrorType.UNKNOWN: UnknownError,
}


def to_gateway_error(exc: BaseException, *, provider: str, status_code: Optional[int] = None) -> GatewayError:
    """Wrap any exception into the appropriate GatewayError subclass.

    Idempotent: if `exc` is already a GatewayError, it is returned unchanged.
    """

    if isinstance(exc, GatewayError):
        return exc

    error_type = classify_exception(exc, status_code=status_code)
    error_cls = _ERROR_CLASS_BY_TYPE[error_type]
    return error_cls(str(exc), provider=provider, cause=exc)
