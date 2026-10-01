"""dict ⇄ runtime vocabulary conversions for the API layer.

The single place HTTP shapes (src/api/schemas.py) meet runtime shapes
(ChatMessage / ToolCall / OrchestrationResult / ExtractionResult). No logic
beyond faithful conversion — every field maps 1:1 so a transcript received
over HTTP can be fed back in and the runtime's transcripts round-trip out
losslessly.

The usage mapper encodes the CLI's token-preference rules (src/cli.py):
provider-billed tokens_in wins; otherwise the caller's client-side
pre-count is used and `token_count_method` labels HOW it was counted.

The inline-schema helper reuses the public schema-pipeline functions —
`validate_json_schema_document()` + `check_supported_subset()` — composed
exactly like `structured.schema.load_and_validate_schema()` minus the file
read (D7: schemas arrive as JSON in the body, never as server-side paths).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..core.errors import SchemaError, UnsupportedSchemaError
from ..core.orchestrator import OrchestrationResult
from ..core.types import ToolCall
from ..providers.base import ChatMessage
from ..structured.extractor import ExtractionResult
from ..structured.schema import check_supported_subset, validate_json_schema_document
from ..token_utils import count_method
from .schemas import (
    ChatResponse,
    MessageIn,
    MessageOut,
    StructuredResponse,
    ToolCallIn,
    ToolCallOut,
    UsageOut,
)


# ---------------------------------------------------------------------------
# Inbound: HTTP request vocabulary → runtime vocabulary
# ---------------------------------------------------------------------------


def tool_call_in_to_tool_call(tool_call: ToolCallIn) -> ToolCall:
    return ToolCall(id=tool_call.id, name=tool_call.name, arguments=tool_call.arguments)


def message_in_to_chat_message(message: MessageIn) -> ChatMessage:
    return ChatMessage(
        role=message.role,
        content=message.content,
        tool_calls=(
            [tool_call_in_to_tool_call(tc) for tc in message.tool_calls] if message.tool_calls else None
        ),
        tool_call_id=message.tool_call_id,
        name=message.name,
    )


def messages_in_to_chat_messages(messages: List[MessageIn]) -> List[ChatMessage]:
    return [message_in_to_chat_message(m) for m in messages]


# ---------------------------------------------------------------------------
# Outbound: runtime vocabulary → HTTP response vocabulary
# ---------------------------------------------------------------------------


def tool_call_to_tool_call_out(tool_call: ToolCall) -> ToolCallOut:
    return ToolCallOut(id=tool_call.id, name=tool_call.name, arguments=tool_call.arguments)


def chat_message_to_message_out(message: ChatMessage) -> MessageOut:
    """Full-fidelity ChatMessage → MessageOut.

    Field vocabulary is identical to core.session's session-record shape
    (_message_to_record): role/content always present, tool_calls /
    tool_call_id / name present only when set. tests/test_api_mappers.py
    cross-checks this against `_message_to_record` itself.
    """
    return MessageOut(
        role=message.role,
        content=message.content,
        tool_calls=(
            [tool_call_to_tool_call_out(tc) for tc in message.tool_calls] if message.tool_calls else None
        ),
        tool_call_id=message.tool_call_id,
        name=message.name,
    )


def build_usage_out(
    provider_tokens_in: Optional[int], pre_count_tokens_in: Optional[int], tokens_out: int
) -> UsageOut:
    """Token accounting with the CLI's preference rules (D12).

    Provider-billed `tokens_in` wins (and carries no method label — the
    count came from the provider, not this client); otherwise the caller's
    client-side pre-count is used and `token_count_method` records the
    counting method ("tiktoken" / "heuristic") for log analysis.
    """
    if provider_tokens_in is not None:
        return UsageOut(tokens_in=provider_tokens_in, tokens_out=tokens_out, token_count_method=None)
    return UsageOut(tokens_in=pre_count_tokens_in, tokens_out=tokens_out, token_count_method=count_method())


def orchestration_result_to_chat_response(
    result: OrchestrationResult,
    *,
    provider: str,
    model: str,
    pre_count_tokens_in: Optional[int],
    warnings: Optional[List[str]] = None,
) -> ChatResponse:
    """One completed turn → ChatResponse.

    `result.messages` is the full transcript (run_turn's continuation
    contract); it is serialized, never mutated. The model name reported is
    the provider's resolved model (`provider.model`), matching what the
    provider actually used — not the raw request field. `warnings`
    carries the handler's non-fatal notes (the stderr analogue).
    """
    return ChatResponse(
        text=result.text,
        data=result.data,
        messages=[chat_message_to_message_out(m) for m in result.messages],
        provider=provider,
        model=model,
        tool_call_count=result.tool_call_count,
        tool_iterations=result.tool_iterations,
        attempts=result.attempts,
        usage=build_usage_out(result.tokens_in, pre_count_tokens_in, result.tokens_out),
        warnings=warnings or [],
    )


def extraction_result_to_structured_response(
    result: ExtractionResult,
    *,
    provider: str,
    model: str,
    pre_count_tokens_in: Optional[int],
) -> StructuredResponse:
    """One completed extraction → StructuredResponse (same usage rules)."""
    return StructuredResponse(
        data=result.data,
        raw_text=result.raw_text,
        attempts=result.attempts,
        provider=provider,
        model=model,
        usage=build_usage_out(result.tokens_in, pre_count_tokens_in, result.tokens_out),
    )


# ---------------------------------------------------------------------------
# Inline schema validation (D7)
# ---------------------------------------------------------------------------


def validate_inline_schema(schema: Any) -> Dict[str, Any]:
    """Validate a schema that arrived inline in a request body (D7).

    Composes the same public pipeline steps as
    `structured.schema.load_and_validate_schema()` minus the file read, so
    inline schemas fail exactly like file schemas would at the CLI:

      1. not a JSON object at the top level          → SchemaError
      2. not structurally valid JSON Schema at all   → SchemaError
      3. root is not "type": "object"                → UnsupportedSchemaError
      4. valid but outside the supported subset      → UnsupportedSchemaError
         (path-qualified, e.g. "$.properties.x: uses keyword(s) ['oneOf'] ...")

    Returns the schema unchanged, ready for `build_model()` /
    `response_schema=` — never mutates or rewrites the client's document.
    """
    if not isinstance(schema, dict):
        raise SchemaError(f"Schema must contain a JSON object at the top level, got {type(schema).__name__}.")

    validate_json_schema_document(schema)

    if schema.get("type") != "object":
        raise UnsupportedSchemaError(
            'Root schema must have "type": "object" — structured extraction '
            "produces one JSON object per call."
        )

    check_supported_subset(schema)
    return schema
