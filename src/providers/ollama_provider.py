"""Ollama provider: talks to a local Ollama server (default http://localhost:11434).

Requires `ollama serve` running locally and the target model pulled
(e.g. `ollama pull llama3.2`). No API key needed — this is the zero-cost,
fully-local half of the gateway.
"""

from __future__ import annotations

import json
import os
from typing import Iterator, List, Optional

import requests

from ..core.errors import FormatError, ModelError, to_gateway_error
from ..token_utils import count_tokens
from .base import BaseProvider, ChatMessage, ChatResponse


class OllamaProvider(BaseProvider):
    name = "ollama"

    def __init__(self, model: str = "llama3.2", base_url: Optional[str] = None, timeout: float = 60.0):
        super().__init__(model)
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    # -- request building ---------------------------------------------------

    def _payload(self, messages: List[ChatMessage], temperature: float, max_tokens: int, stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

    def _post(self, payload: dict, *, stream: bool) -> requests.Response:
        url = f"{self.base_url}/api/chat"
        try:
            return requests.post(url, json=payload, timeout=self.timeout, stream=stream)
        except requests.exceptions.ConnectionError as exc:
            raise ModelError(
                f"Could not reach Ollama at {self.base_url}. Is `ollama serve` running?",
                provider=self.name,
                cause=exc,
            )
        except requests.exceptions.Timeout as exc:
            raise ModelError("Ollama request timed out.", provider=self.name, cause=exc)

    # -- public API -----------------------------------------------------------

    def chat(self, messages, *, temperature: float = 0.7, max_tokens: int = 512) -> ChatResponse:
        resp = self._post(self._payload(messages, temperature, max_tokens, stream=False), stream=False)

        if resp.status_code != 200:
            raise to_gateway_error(RuntimeError(resp.text), provider=self.name, status_code=resp.status_code)

        try:
            data = resp.json()
            text = data["message"]["content"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise FormatError(f"Unexpected Ollama response shape: {exc}", provider=self.name, cause=exc)

        return ChatResponse(text=text, tokens_out=count_tokens(text), raw=data)

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
