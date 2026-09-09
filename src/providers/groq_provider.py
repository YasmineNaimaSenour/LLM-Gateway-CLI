"""Groq provider: OpenAI-compatible chat completions API.

Requires GROQ_API_KEY. This is the hosted, faster half of the gateway.

Tool calling: Groq's `tool_calls` include a real `id`, but
`function.arguments` comes back as a JSON-*encoded string*, not a dict —
this adapter parses it before handing a normalized ToolCall to gateway-core.

Structured outputs: Groq/OpenAI-compatible chat completions don't offer a
strict schema-constrained decoding mode for these models, only a looser
`response_format: json_object` (valid-JSON-syntax, not schema-conforming).
We use it as a best-effort reliability boost; gateway-core's validation and
retry loop is what actually enforces the schema either way.
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

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

@register_provider("groq", default_model="openai/gpt-oss-20b")
class GroqProvider(BaseProvider):
    name = "groq"

    def __init__(self, model: str = "openai/gpt-oss-20b", api_key: Optional[str] = None, timeout: float = 60.0):
        super().__init__(model)
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.timeout = timeout
        if not self.api_key:
            raise ModelError(
                "GROQ_API_KEY is not set. Export it or pass api_key explicitly.",
                provider=self.name,
            )

    # -- request building -----------------------------------------------------

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    @staticmethod
    def _message_payload(m: ChatMessage) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"role": m.role, "content": m.content or ""}
        if m.role == "assistant" and m.tool_calls:
            payload["content"] = m.content  # OpenAI-style: null/empty content alongside tool_calls is normal
            payload["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in m.tool_calls
            ]
        if m.role == "tool":
            payload["tool_call_id"] = m.tool_call_id or ""
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
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if tools:
            payload["tools"] = self._tools_payload(tools)
        if response_schema is not None:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _post(self, payload: dict, *, stream: bool) -> requests.Response:
        try:
            # post_with_retry owns the should-I-retry decision for transient
            # failures (connection blips, 5xx); exhaustion re-raises, and the
            # handlers below stay responsible for how the failure is presented.
            return post_with_retry(
                GROQ_API_URL,
                payload=payload,
                headers=self._headers(),
                timeout=self.timeout,
                stream=stream,
            )
        except requests.exceptions.ConnectionError as exc:
            raise ModelError("Could not reach the Groq API. Check your network connection.", provider=self.name, cause=exc)
        except requests.exceptions.Timeout as exc:
            raise ModelError("Groq request timed out.", provider=self.name, cause=exc)

    @staticmethod
    def _error_message(resp: requests.Response) -> str:
        try:
            body = resp.json()
            return body.get("error", {}).get("message", resp.text)
        except json.JSONDecodeError:
            return resp.text

    def _parse_tool_calls(self, message: dict) -> List[ToolCall]:
        raw_calls = message.get("tool_calls") or []
        calls: List[ToolCall] = []
        for raw in raw_calls:
            fn = raw.get("function", {})
            raw_args = fn.get("arguments", "{}")
            try:
                arguments = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except json.JSONDecodeError as exc:
                raise FormatError(f"Malformed tool-call arguments from Groq: {exc}", provider=self.name, cause=exc)
            calls.append(ToolCall(id=raw.get("id", ""), name=fn.get("name", ""), arguments=arguments))
        return calls

    # -- public API -------------------------------------------------------------

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
            raise to_gateway_error(RuntimeError(self._error_message(resp)), provider=self.name, status_code=resp.status_code)

        try:
            data = resp.json()
            message = data["choices"][0]["message"]
            text = message.get("content") or ""
            usage = data.get("usage", {})
            tokens_out = usage.get("completion_tokens") or count_tokens(text)
            tokens_in = usage.get("prompt_tokens")  # provider-billed count (audit #11); None if unreported
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise FormatError(f"Unexpected Groq response shape: {exc}", provider=self.name, cause=exc)

        tool_calls = self._parse_tool_calls(message)
        return ChatResponse(text=text, tokens_out=tokens_out, tool_calls=tool_calls, raw=data, tokens_in=tokens_in)

    def chat_stream(self, messages, *, temperature: float = 0.7, max_tokens: int = 512) -> Iterator[str]:
        resp = self._post(self._payload(messages, temperature, max_tokens, stream=True), stream=True)

        if resp.status_code != 200:
            raise to_gateway_error(RuntimeError(self._error_message(resp)), provider=self.name, status_code=resp.status_code)

        for line in resp.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8") if isinstance(line, bytes) else line
            if not decoded.startswith("data:"):
                continue
            payload = decoded[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
                delta = chunk["choices"][0]["delta"].get("content", "")
            except (json.JSONDecodeError, KeyError, IndexError) as exc:
                raise FormatError(f"Malformed streaming chunk from Groq: {exc}", provider=self.name, cause=exc)
            if delta:
                yield delta