"""Groq provider: OpenAI-compatible chat completions API.

Requires GROQ_API_KEY. This is the hosted, faster half of the gateway.
"""

from __future__ import annotations

import json
import os
from typing import Iterator, List, Optional

import requests

from ..core.errors import FormatError, ModelError, to_gateway_error
from ..token_utils import count_tokens
from .base import BaseProvider, ChatMessage, ChatResponse

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqProvider(BaseProvider):
    name = "groq"

    def __init__(self, model: str = "llama-3.1-8b-instant", api_key: Optional[str] = None, timeout: float = 60.0):
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

    def _payload(self, messages: List[ChatMessage], temperature: float, max_tokens: int, stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

    def _post(self, payload: dict, *, stream: bool) -> requests.Response:
        try:
            return requests.post(GROQ_API_URL, headers=self._headers(), json=payload, timeout=self.timeout, stream=stream)
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

    # -- public API -------------------------------------------------------------

    def chat(self, messages, *, temperature: float = 0.7, max_tokens: int = 512) -> ChatResponse:
        resp = self._post(self._payload(messages, temperature, max_tokens, stream=False), stream=False)

        if resp.status_code != 200:
            raise to_gateway_error(RuntimeError(self._error_message(resp)), provider=self.name, status_code=resp.status_code)

        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            tokens_out = usage.get("completion_tokens") or count_tokens(text)
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise FormatError(f"Unexpected Groq response shape: {exc}", provider=self.name, cause=exc)

        return ChatResponse(text=text, tokens_out=tokens_out, raw=data)

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
