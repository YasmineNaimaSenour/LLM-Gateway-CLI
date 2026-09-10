"""Ollama provider: talks to a local Ollama server (default http://localhost:11434).

Requires `ollama serve` running locally and the target model pulled
(e.g. `ollama pull llama3.2`). No API key needed — this is the zero-cost,
fully-local half of the gateway.

Tool calling: Ollama's native /api/chat accepts an OpenAI-style `tools`
list and returns `tool_calls` on the response message, but (unlike
OpenAI/Groq) does not guarantee a stable per-call id — we synthesize one.
Tool-call arguments come back already parsed as a dict, no json.loads needed.

Structured outputs: Ollama supports passing a full JSON Schema into the
`format` field for grammar-constrained decoding, which is why
`response_schema` is forwarded there directly rather than emulated purely
through prompting.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, List, Optional

from dotenv import load_dotenv

import requests

from ..core.errors import FormatError, ModelError, to_gateway_error
from ..core.types import ToolCall, ToolSpec
from ..token_utils import count_tokens
from .base import BaseProvider, ChatMessage, ChatResponse
from .http_utils import post_with_retry
from .registry import register_provider

load_dotenv()

# Interactive-friendly default (audit #23): local models are usually fast,
# and a stuck server should surface as an error in ~a minute, not two. Batch
# users or slow hardware can raise it via OLLAMA_TIMEOUT — the env var, not
# the constructor default, is the documented override.
DEFAULT_OLLAMA_TIMEOUT = 60.0


@register_provider("ollama", default_model="llama3.2")
class OllamaProvider(BaseProvider):
    name = "ollama"

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        super().__init__(model)
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        # Precedence: explicit argument > OLLAMA_TIMEOUT env var > 60s default.
        env_timeout = os.environ.get("OLLAMA_TIMEOUT")
        if timeout is not None:
            self.timeout = float(timeout)
        elif env_timeout:
            self.timeout = float(env_timeout)
        else:
            self.timeout = DEFAULT_OLLAMA_TIMEOUT

    # -- request building ---------------------------------------------------

    @staticmethod
    def _message_payload(m: ChatMessage) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"role": m.role, "content": m.content or ""}
        if m.role == "assistant" and m.tool_calls:
            payload["tool_calls"] = [
                {"function": {"name": tc.name, "arguments": tc.arguments}} for tc in m.tool_calls
            ]
        if m.role == "tool":
            payload["tool_name"] = m.name or ""
        return payload

    @staticmethod
    def _tools_payload(tools: List[ToolSpec]) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
            }
            for t in tools
        ]

    def _payload(
        self,
        messages: List[ChatMessage],
        temperature: float,
        max_tokens: int,
        stream: bool,
        tools: Optional[List[ToolSpec]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> dict:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [self._message_payload(m) for m in messages],
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if tools:
            payload["tools"] = self._tools_payload(tools)
        if response_schema is not None:
            payload["format"] = response_schema
        return payload

    def _post(self, payload: dict, *, stream: bool) -> requests.Response:
        url = f"{self.base_url}/api/chat"
        try:
            # post_with_retry owns the should-I-retry decision for transient
            # failures (connection blips, 5xx); exhaustion re-raises, and the
            # handlers below stay responsible for how the failure is presented.
            return post_with_retry(url, payload=payload, timeout=self.timeout, stream=stream)
        except requests.exceptions.ConnectionError as exc:
            raise ModelError(
                f"Could not reach Ollama at {self.base_url}. Is `ollama serve` running?",
                provider=self.name,
                cause=exc,
            )
        except requests.exceptions.Timeout as exc:
            raise ModelError("Ollama request timed out.", provider=self.name, cause=exc)

    @staticmethod
    def _parse_tool_calls(message: dict) -> List[ToolCall]:
        raw_calls = message.get("tool_calls") or []
        calls: List[ToolCall] = []
        for i, raw in enumerate(raw_calls):
            fn = raw.get("function", {})
            calls.append(
                ToolCall(
                    id=raw.get("id") or f"call_{i}",  # Ollama's native format doesn't guarantee an id
                    name=fn.get("name", ""),
                    arguments=fn.get("arguments") or {},
                )
            )
        return calls

    # -- public API -----------------------------------------------------------

    def chat(
        self,
        messages,
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        tools: Optional[List[ToolSpec]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> ChatResponse:
        resp = self._post(
            self._payload(messages, temperature, max_tokens, stream=False, tools=tools, response_schema=response_schema),
            stream=False,
        )

        if resp.status_code != 200:
            raise to_gateway_error(RuntimeError(resp.text), provider=self.name, status_code=resp.status_code)

        try:
            data = resp.json()
            message = data["message"]
            text = message.get("content") or ""
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise FormatError(f"Unexpected Ollama response shape: {exc}", provider=self.name, cause=exc)

        tool_calls = self._parse_tool_calls(message)
        # Ollama reports prompt usage on the final non-streaming chunk;
        # surface tokens_in when present (audit #11), else None so callers
        # fall back to client-side counting.
        prompt_eval_count = data.get("prompt_eval_count")
        return ChatResponse(
            text=text,
            tokens_out=count_tokens(text),
            tool_calls=tool_calls,
            raw=data,
            tokens_in=prompt_eval_count if isinstance(prompt_eval_count, int) else None,
        )

    def chat_stream(self, messages, *, temperature: float = 0.7, max_tokens: int = 512) -> Iterator[str]:
        resp = self._post(self._payload(messages, temperature, max_tokens, stream=True), stream=True)

        if resp.status_code != 200:
            raise to_gateway_error(RuntimeError(resp.text), provider=self.name, status_code=resp.status_code)

        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FormatError(f"Malformed streaming chunk from Ollama: {exc}", provider=self.name, cause=exc)

            if chunk.get("error"):
                raise to_gateway_error(RuntimeError(chunk["error"]), provider=self.name)

            piece = chunk.get("message", {}).get("content", "")
            if piece:
                yield piece
            if chunk.get("done"):
                break