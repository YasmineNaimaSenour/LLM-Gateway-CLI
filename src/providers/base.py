"""Provider-agnostic interface every LLM backend must implement.

Any new backend (OpenAI, Anthropic, local llama.cpp, ...) only needs to
subclass BaseProvider and implement chat() / chat_stream(). The CLI and
logging layer never need to know which provider is behind the interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, List, Optional


@dataclass
class ChatMessage:
    """A single message in the conversation."""

    role: str  # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResponse:
    """Final, fully-assembled response (used only for both the sync path)."""

    text: str
    tokens_out: int
    raw: Optional[dict] = None


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
    ) -> ChatResponse:
        """Non-streaming call. Blocks until the full response is available."""
        raise NotImplementedError

    @abstractmethod
    def chat_stream(
        self,
        messages: List[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Iterator[str]:
        """Streaming call. Yields text chunks as they arrive from the provider."""
        raise NotImplementedError
