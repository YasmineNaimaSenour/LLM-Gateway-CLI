"""HTTP-facing request/response models for the API layer.

Pydantic is used here ONLY as HTTP DTO validation — a deliberate detail of
src/api/ unrelated to src/structured's internal schema→model machinery.
The vocabulary mirrors the runtime exactly (ChatMessage / ToolCall /
OrchestrationResult / ExtractionResult) so mappers.py can convert
1:1 with no reshaping and no information loss.

No business logic: two cross-field validators guard the request contract
(one message form per request; streaming is plain chat only), everything
else is shape + defaults matching the CLI's (D5, D8).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..core.orchestrator import DEFAULT_MAX_TOOL_ITERATIONS
from ..structured.extractor import DEFAULT_MAX_RETRIES

_ROLES = Literal["system", "user", "assistant", "tool"]


# ---------------------------------------------------------------------------
# Request models (§3.2)
# ---------------------------------------------------------------------------


class ToolCallIn(BaseModel):
    """HTTP form of core.types.ToolCall — shape validation only, no logic."""

    id: str
    name: str
    arguments: Dict[str, Any]


class MessageIn(BaseModel):
    """HTTP form of providers.base.ChatMessage.

    Accepts the full-fidelity shape a session record uses (tool_calls on
    assistant messages, tool_call_id/name on tool messages), so clients can
    round-trip transcripts received from /v1/chat straight back in.
    """

    role: _ROLES
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCallIn]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatRequest(BaseModel):
    """Body of POST /v1/chat.

    Two message forms, exactly one per request (D5):
      - `messages` — full-history form: the client holds the transcript
        (the primary mechanism; matches run_turn's transcript contract).
      - `system` + `prompt` — convenience form (optionally with
        `session_path` for server-side history).

    Cross-field rule (D8): `stream: true` is rejected when `tools` or
    `schema` is present — chat_stream() is text-only and the API fails
    loudly instead of silently downgrading a contract.
    """

    model_config = ConfigDict(populate_by_name=True)

    provider: str
    model: Optional[str] = None  # None → registry default (per provider)
    timeout: Optional[float] = None  # forwarded to registry, like --timeout
    messages: Optional[List[MessageIn]] = None  # full-history form
    system: Optional[str] = None  # convenience form
    prompt: Optional[str] = None  # convenience form
    temperature: float = 0.7  # CLI chat default
    max_tokens: int = 512  # CLI default
    stream: bool = False
    tools: Optional[List[str]] = None  # names resolved per request (D13)
    max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS
    schema_: Optional[Dict[str, Any]] = Field(default=None, alias="schema")  # inline schema (D7)
    max_retries: int = DEFAULT_MAX_RETRIES
    session_path: Optional[str] = None  # server-side JSONL transcript (D9)

    @field_validator("messages")
    @classmethod
    def _messages_not_empty(cls, v: Optional[List[MessageIn]]) -> Optional[List[MessageIn]]:
        if v is not None and not v:
            raise ValueError("messages must not be an empty list (use `prompt` for a single-turn request).")
        return v

    @model_validator(mode="after")
    def _validate_message_forms(self) -> "ChatRequest":
        has_messages = self.messages is not None
        has_prompt = self.prompt is not None
        if has_messages and has_prompt:
            raise ValueError(
                "Provide either `messages` (full history) or `prompt` (convenience form), not both."
            )
        if not has_messages and not has_prompt:
            raise ValueError(
                "One of `messages` (full history) or `prompt` (convenience form) is required."
            )
        return self

    @model_validator(mode="after")
    def _validate_stream_is_plain_chat(self) -> "ChatRequest":
        if self.stream and (self.tools or self.schema_ is not None):
            conflicts = []
            if self.tools:
                conflicts.append("`tools`")
            if self.schema_ is not None:
                conflicts.append("`schema`")
            raise ValueError(
                "stream=true cannot be combined with " + " or ".join(conflicts)
                + ": streaming is plain chat only (text chunks); "
                "tool-bearing turns are non-streaming by contract and schema "
                "coercion replaces the streamed text."
            )
        return self


class ChatStreamRequest(ChatRequest):
    """Body of POST /v1/chat/stream: plain chat only (D8).

    Subclass, not duplication: identical fields and defaults, with `stream`
    pinned to True. That pin makes the inherited D8 cross-field validator
    reject `tools`/`schema` unconditionally on this endpoint — the 422 fires
    at body validation, before the response (and its SSE headers) starts.
    The stream flag itself is not repeated in client bodies; it is what this
    endpoint *is*.
    """

    stream: Literal[True] = True  # pinned: the endpoint is the stream request


class StructuredRequest(BaseModel):
    """Body of POST /v1/structured: text + inline schema → validated JSON.

    Mirrors `structured --input file --schema file`; over HTTP the input is
    the `text` field and the schema is inline in the body (D7 — no
    server-side paths). Temperature defaults to 0.0 like the CLI's
    structured command (determinism by default).
    """

    model_config = ConfigDict(populate_by_name=True)

    provider: str
    model: Optional[str] = None
    timeout: Optional[float] = None
    text: str  # HTTP form of the CLI's --input file
    schema_: Dict[str, Any] = Field(alias="schema")
    temperature: float = 0.0  # CLI structured default: determinism
    max_tokens: int = 512
    max_retries: int = DEFAULT_MAX_RETRIES


# ---------------------------------------------------------------------------
# Response models (§3.3)
# ---------------------------------------------------------------------------


class ToolCallOut(BaseModel):
    """One tool invocation requested by the model, as it appears in transcripts."""

    id: str
    name: str
    arguments: Dict[str, Any]


class MessageOut(BaseModel):
    """One transcript message, full-fidelity — same fields as
    providers.base.ChatMessage, so transcripts received from /v1/chat can
    be fed straight back in as `messages` (the continuation contract).

    The session-JSONL record shape (core.session._message_to_record: unset
    optional fields omitted, `content` kept even when None) is produced by
    mappers.py, which owns the ChatMessage/MessageOut ⇄ record
    conversion and cross-checks it against `_message_to_record`.
    """

    role: _ROLES
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCallOut]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class UsageOut(BaseModel):
    """Token accounting with the CLI's preference rules (D12).

    `tokens_in` is the provider-billed count when available; when None,
    `token_count_method` names the client-side fallback method
    ("tiktoken" / "heuristic") that produced the counts.
    """

    tokens_in: Optional[int] = None  # provider-billed, else client-side pre-count
    tokens_out: int
    token_count_method: Optional[str] = None


class ChatResponse(BaseModel):
    """One completed chat turn: the final text, optional validated data,
    the full transcript (continuation contract), and usage/tool stats.

    `warnings` carries non-fatal notes (the over-HTTP channel for
    what the CLI prints to stderr, e.g. a session save that failed after
    a successful turn); empty when nothing noteworthy happened.
    """

    text: str
    data: Optional[Dict[str, Any]] = None  # validated JSON when schema was given
    messages: List[MessageOut]  # full transcript — feed back via `messages`
    provider: str
    model: str
    tool_call_count: int
    tool_iterations: int
    attempts: int
    usage: UsageOut
    warnings: List[str] = Field(default_factory=list)


class StructuredResponse(BaseModel):
    """One completed structured extraction."""

    data: Dict[str, Any]
    raw_text: str
    attempts: int
    provider: str
    model: str
    usage: UsageOut
