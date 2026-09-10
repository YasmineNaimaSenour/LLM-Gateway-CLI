"""Session persistence: continue a conversation across CLI invocations.

This is the "multi-turn / session concept" the CLI was missing (AUDIT.md
#2). The orchestrator already returns the complete transcript for a turn
(`OrchestrationResult.messages`, AUDIT.md #1/#19) — a session is simply
that transcript, loaded from and appended to a JSONL file:

    {"role": ..., "content": ..., "tool_calls": [...], "tool_call_id": ..., "name": ...}

One line per ChatMessage, appended in order. That makes the session file
the conversation history itself rather than a separate index, which means:

* state is inspectable with `cat`/`jq` and recoverable if the process dies
  mid-write (a torn trailing line is skipped on load, never fatal),
* continuing a session is load-append with no read-modify-write window,
  and
* the on-disk format matches the orchestrator's in-memory vocabulary 1:1,
  so there is no translation layer to keep in sync.

Sessions store full-fidelity messages — including assistant tool_calls and
tool results — so a tool-calling conversation resumes with its full
context, not just the prose. System/schema messages are stored too, so the
saved session replays exactly what the model saw.

Why JSONL rather than one JSON file? Append-only writes never truncate
prior state (crash-safe), the history is diffable, and it mirrors
logs/requests.jsonl — the project already treats "one JSON object per
line" as its persistence idiom.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.errors import SessionError
from ..core.types import ToolCall
from ..providers.base import ChatMessage

VALID_ROLES = ("system", "user", "assistant", "tool")


def _message_to_record(message: ChatMessage) -> Dict[str, Any]:
    """Full-fidelity ChatMessage -> JSON-serializable dict.

    (ChatMessage.to_content_dict() is deliberately content-only — audit #22 —
    so sessions serialize explicitly rather than relying on it.)
    """
    record: Dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        record["tool_calls"] = [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in message.tool_calls
        ]
    if message.tool_call_id is not None:
        record["tool_call_id"] = message.tool_call_id
    if message.name is not None:
        record["name"] = message.name
    return record


def _message_from_record(record: Dict[str, Any]) -> Optional[ChatMessage]:
    """Parse one JSONL record into a ChatMessage, or None if it's not a
    valid session record (unknown role, malformed tool_calls, ...)."""
    role = record.get("role")
    if role not in VALID_ROLES:
        return None
    tool_calls: Optional[List[ToolCall]] = None
    if record.get("tool_calls"):
        calls = []
        for tc in record["tool_calls"]:
            arguments = tc.get("arguments")
            if not isinstance(arguments, dict):
                return None
            calls.append(ToolCall(id=tc["id"], name=tc["name"], arguments=arguments))
        tool_calls = calls
    return ChatMessage(
        role=role,
        content=record.get("content"),
        tool_calls=tool_calls,
        tool_call_id=record.get("tool_call_id"),
        name=record.get("name"),
    )


class SessionStore:
    """Append-only JSONL persistence for a conversation transcript.

    One SessionStore per session file. `load()` returns every message on
    disk; `append()` writes new lines at the end. Nothing is ever rewritten
    or truncated.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> List[ChatMessage]:
        """Return all messages persisted in the session file.

        A torn or malformed line (e.g. the process died mid-append) is
        skipped rather than fatal. Raises SessionError only if the file
        contains no valid session records at all — that's almost certainly
        not a session file, and silently starting fresh would discard the
        user's intent.
        """
        if not self.exists():
            return []
        messages: List[ChatMessage] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    message = _message_from_record(json.loads(line))
                except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                    continue
                if message is not None:
                    messages.append(message)
        if not messages:
            raise SessionError(
                f"Session file {self.path} contains no valid session records; refusing to treat it as a session."
            )
        return messages

    def append(self, messages: List[ChatMessage]) -> None:
        """Append messages to the session file, creating it (and parent
        directories) on first write."""
        if not messages:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            for message in messages:
                f.write(json.dumps(_message_to_record(message), ensure_ascii=False) + "\n")


def load_session_messages(path: str) -> Optional[List[ChatMessage]]:
    """CLI helper: load prior history for a --session file, if one exists.

    Returns None when no session file exists yet (the caller starts fresh;
    the file is created on save). Raises SessionError if the path exists
    but isn't a usable session file.
    """
    store = SessionStore(path)
    if not store.exists():
        return None
    return store.load()


def save_session_messages(path: str, messages: List[ChatMessage]) -> None:
    """CLI helper: persist a completed turn's transcript to a --session file.

    The transcript is the full conversation state (system + user + assistant
    + tool messages). The file already holds every message from earlier
    turns, so only the tail after the common prefix with what's on disk is
    appended — calling this twice with the same transcript is a no-op, and
    a continuation turn writes just its new messages.
    """
    store = SessionStore(path)
    existing = store.load() if store.exists() else []
    prefix_len = 0
    for saved, new in zip(existing, messages):
        if saved != new:
            break
        prefix_len += 1
    store.append(messages[prefix_len:])
