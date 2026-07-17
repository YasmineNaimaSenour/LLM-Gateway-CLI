"""Token counting utilities.

Counts tokens *before* a request is sent, so the gateway can log tokens_in
and (upstream) decide whether a prompt is likely to blow the context window.

Uses tiktoken's cl100k_base encoding when available. This is an approximation
for non-OpenAI models (Llama, etc.) but is close enough for logging and
budget-tracking purposes. Falls back to a word-count heuristic so the
gateway keeps working in fully offline / dependency-light environments.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping

try:
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - exercised only when tiktoken is absent
    _ENCODING = None

_PER_MESSAGE_OVERHEAD = 4  # rough allowance for role/formatting tokens per chat message


def count_tokens(text: str) -> int:
    """Return an (approximate) token count for a single string."""
    if not text:
        return 0
    if _ENCODING is not None:
        return len(_ENCODING.encode(text))
    return _heuristic_token_count(text)


def _heuristic_token_count(text: str) -> int:
    """Fallback estimator: ~0.75 tokens per whitespace-delimited word (English rule of thumb)."""
    words = re.findall(r"\S+", text)
    return max(1, round(len(words) / 0.75))


def count_message_tokens(messages: Iterable[Mapping[str, str]]) -> int:
    """Return the total approximate token count across a list of chat messages."""
    total = 0
    for message in messages:
        total += count_tokens(message.get("content", ""))
        total += _PER_MESSAGE_OVERHEAD
    return total
