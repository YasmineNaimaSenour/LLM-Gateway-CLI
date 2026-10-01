"""SSE streaming bridge: provider.chat_stream() → text/event-stream (§2.3, §3.5).

A sync generator (D4: Starlette iterates sync generators in a threadpool)
that owns the stream's *choreography* — the HTTP analogue of the CLI's
`_run_stream()` + `_main_chat()` tail:

  - every chunk  → one `data: {"delta": "..."}` frame, accumulated;
  - natural end  → append the assistant message, session save (after
    success only, warn-not-fail → `warnings` on the done event), success
    `log_request` with the CLI's `_run_stream` rules — streaming never
    gets provider-billed counts: `tokens_out = count_tokens(joined)`,
    `tokens_in = the route's client-side pre-count`,
    `token_count_method = count_method()` — then the `done` event and the
    `[DONE]` sentinel;
  - any error    → error `log_request` (via error_handlers' field logic —
    the `_log_and_report` analogue), one `data: {"error": {...}}` event in
    the uniform envelope, then `[DONE]` and the stream closes.

Error handling lives *inside* the generator because SSE has no HTTP status
after the headers are sent: anything the provider raises mid-stream (or
before the first chunk — the response has still started by then) becomes
an error event, never a broken connection with no explanation. Request-
shaped failures (validation, deps) raise in the route *before* the
StreamingResponse exists and keep normal HTTP statuses.
"""

from __future__ import annotations

import json
from typing import Iterator, List, Optional

from ..core.errors import GatewayError, to_gateway_error
from ..core.logger import log_request
from ..core.session import save_session_messages
from ..core.telemetry import Timer
from ..providers.base import BaseProvider, ChatMessage
from ..token_utils import count_method, count_tokens
from .error_handlers import build_error_body, log_gateway_error

# Sentinel closing every stream, mirroring the project's Groq provider idiom.
_DONE_SENTINEL = "[DONE]"


def _log_kwargs(log_path) -> dict:
    # Same None-omission pattern as app.py / error_handlers: log_path=None
    # means the logger's own (env-coupled) default.
    return {"log_path": log_path} if log_path is not None else {}


def sse_event(payload: dict) -> str:
    """One SSE frame. json.dumps never emits raw newlines, so a frame is
    always exactly one `data:` line followed by the blank-line separator."""
    return f"data: {json.dumps(payload)}\n\n"


def done_sentinel() -> str:
    return f"data: {_DONE_SENTINEL}\n\n"


def sse_chat_stream(
    provider: BaseProvider,
    messages: List[ChatMessage],
    *,
    temperature: float,
    max_tokens: int,
    pre_count_tokens_in: int,
    timer: Timer,
    session_path: Optional[str] = None,
    log_path=None,
    warnings: Optional[List[str]] = None,
) -> Iterator[str]:
    """Yield SSE frames for one plain-chat streaming turn.

    `messages` is the fully assembled input (the route's job, identical to
    /v1/chat); `pre_count_tokens_in` is the route's client-side pre-count,
    made before the response started. `timer` starts in the route so the
    success/error record's latency covers the whole streamed turn.
    `warnings` carries the route's continuation notes (the session
    guard's) — surfaced on the done event alongside any save warnings.
    """
    warnings = list(warnings or [])
    chunks: List[str] = []

    try:
        for chunk in provider.chat_stream(messages, temperature=temperature, max_tokens=max_tokens):
            chunks.append(chunk)
            yield sse_event({"delta": chunk})
    except GatewayError as exc:
        timer.stop()
        log_gateway_error(
            exc, latency_ms=timer.elapsed_ms, temperature=temperature, log_path=log_path
        )
        yield sse_event(build_error_body(exc))
        yield done_sentinel()
        return
    except Exception as exc:  # the stream must close with an explanation, never a broken pipe
        gateway_exc = to_gateway_error(exc, provider=provider.name)
        timer.stop()
        log_gateway_error(
            gateway_exc, latency_ms=timer.elapsed_ms, temperature=temperature, log_path=log_path
        )
        yield sse_event(build_error_body(gateway_exc))
        yield done_sentinel()
        return

    # Natural end — the CLI's `_run_stream` return rules, verbatim:
    # chunks carry no usage report, so there is no provider-billed tokens_in.
    text = "".join(chunks)
    transcript = list(messages) + [ChatMessage(role="assistant", content=text or None)]

    # Session save — after success only (contract #5); a failed save must
    # not fail a completed stream, so it degrades to a warning on the done
    # event (the stderr analogue).
    if session_path:
        try:
            save_session_messages(session_path, transcript)
        except Exception as exc:  # noqa: BLE001  # the answer already streamed
            warnings.append(f"Could not update session file {session_path}: {exc}")

    tokens_out = count_tokens(text)
    timer.stop()
    log_request(
        provider=provider.name,
        latency_ms=timer.elapsed_ms,
        tokens_in=pre_count_tokens_in,
        tokens_out=tokens_out,
        temperature=temperature,
        status="success",
        error_type=None,
        token_count_method=count_method(),
        **_log_kwargs(log_path),
    )

    yield sse_event(
        {
            "done": True,
            "text": text,
            "usage": {
                "tokens_in": pre_count_tokens_in,
                "tokens_out": tokens_out,
                "token_count_method": count_method(),
            },
            "warnings": warnings,
        }
    )
    yield done_sentinel()
