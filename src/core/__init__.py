"""Core runtime: error taxonomy, structured logging, orchestration, sessions.

This subpackage owns the gateway's shared contracts:

- errors      — the GatewayError hierarchy and the 5-category ErrorType
                taxonomy every failure is classified into (the primary
                debugging signal)
- logger      — the JSONL request log written for every call
- orchestrator— run_turn(): the tool-call loop and schema coercion that
                both CLI commands are thin wrappers over
- session     — multi-turn persistence behind `chat --session`
- telemetry   — latency timing helpers
- types       — wire-facing dataclasses shared with providers/tools
"""

from .errors import (
    ContextOverflowError,
    ErrorType,
    ExtractionError,
    FormatError,
    GatewayError,
    ModelError,
    RateLimitError,
    SchemaError,
    SessionError,
    ToolLoopError,
    UnknownError,
    UnsupportedSchemaError,
    to_gateway_error,
)
from .logger import DEFAULT_LOG_PATH, log_request
from .orchestrator import (
    DEFAULT_MAX_TOOL_ITERATIONS,
    OrchestrationResult,
    ToolLoopObserver,
    run_turn,
)
from .session import SessionStore, load_session_messages, save_session_messages
from .telemetry import Timer, measure_latency
from .types import ToolCall, ToolResult, ToolSpec

__all__ = [
    # errors — the taxonomy
    "ErrorType",
    "GatewayError",
    "RateLimitError",
    "ContextOverflowError",
    "FormatError",
    "SchemaError",
    "UnsupportedSchemaError",
    "ExtractionError",
    "ToolLoopError",
    "SessionError",
    "ModelError",
    "UnknownError",
    "to_gateway_error",
    # logger
    "DEFAULT_LOG_PATH",
    "log_request",
    # orchestrator
    "DEFAULT_MAX_TOOL_ITERATIONS",
    "OrchestrationResult",
    "ToolLoopObserver",
    "run_turn",
    # session
    "SessionStore",
    "load_session_messages",
    "save_session_messages",
    # telemetry
    "Timer",
    "measure_latency",
    # types
    "ToolSpec",
    "ToolCall",
    "ToolResult",
]
