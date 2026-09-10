"""Provider-agnostic interface every LLM backend must implement.

Any new backend (OpenAI, Anthropic, local llama.cpp, ...) only needs to
subclass BaseProvider and implement chat() / chat_stream(). The CLI and
logging layer never need to know which provider is behind the interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Literal

from ..core.types import ToolCall, ToolSpec


@dataclass
class ChatMessage:
    """A single message in the conversation.

    `tool_calls` is only ever set on assistant messages (the model
    requesting tool invocations). `tool_call_id` and `name` are only ever
    set on "tool" messages (a tool's result being reported back — id ties
    it to the request, name is the tool's name, needed by providers whose
    wire format identifies tool results by name rather than by id).
    """

    role: Literal["system", "user", "assistant", "tool"]  # "system" | "user" | "assistant" | "tool"
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    def to_content_dict(self) -> dict:
        """Serialize the message's role and text content only.

        Deliberately named for what it does (audit #22): the tool-calling
        fields — `tool_calls`, `tool_call_id`, `name` — are NOT included.
        Token counting is the intended consumer (content is all that
        matters there). For full-fidelity serialization (sessions,
        transcripts) use the explicit record serializers instead — e.g.
        src/core/session.py's, which round-trips every field.
        """
        return {"role": self.role, "content": self.content or ""}


@dataclass
class ChatResponse:
    """Final, fully-assembled response for one non-streaming call.

    `tokens_in` is the provider's own count of the prompt tokens it billed
    (audit #11), or None when the provider doesn't report usage — callers
    then fall back to client-side counting. `tokens_out` keeps the same
    fallback behavior locally, since it has always been required.
    """

    text: str
    tokens_out: int
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw: Optional[dict] = None
    tokens_in: Optional[int] = None


class BaseProvider(ABC):
    """Common contract for all providers (Ollama, Groq, ...)."""

    name: str = "base"

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def chat(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        tools: Optional[List[ToolSpec]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> ChatResponse:
        """Non-streaming call. Blocks until the full response is available.

        `tools`, when given, are offered to the model; a resulting
        ChatResponse.tool_calls is empty when the model didn't call any.
        `response_schema` is a best-effort hint: providers that support
        native schema-constrained or JSON-mode decoding should use it to
        improve reliability, but callers must still validate the result
        themselves — this is never a substitute for that.
        """
        raise NotImplementedError

    @abstractmethod
    def chat_stream(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Iterator[str]:
        """Streaming call. Yields text chunks as they arrive from the provider.

        Deliberately has no `tools` parameter: tool-bearing turns are always
        non-streaming (see core/orchestrator.py) since the two providers
        stream tool-call payloads in incompatible shapes, and there is
        little user-facing value in streaming a turn that may not even
        contain the final answer.
        """
        raise NotImplementedError