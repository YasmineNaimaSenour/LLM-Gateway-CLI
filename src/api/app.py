"""FastAPI application factory: the HTTP routes over the runtime (§3.1).

``create_app()`` is the only constructor; the module-level
``app = create_app()`` exists solely so ``uvicorn src.api.app:app`` works
with zero flags (D3).

Every route is a plain ``def`` (D4): the provider contract is synchronous
(``requests``-based ``chat()``/``chat_stream()``), so Starlette runs the
handlers — and the blocking provider calls inside them — on the threadpool
instead of stalling the event loop.

The handlers are pure choreography (D1) — the exact wiring ``src/cli.py``
performs for its commands: resolve (schema → tools → session → provider,
all pre-network), assemble messages, client-side pre-count, one runtime
call, session save after success only, one success log record built with
the CLI's field rules, map → 200. They contain no business logic and **no**
per-route error handling: any ``GatewayError`` (or unexpected exception)
propagates to the single handler set registered by
``error_handlers.register_handlers`` (D6) — the direct analogue of the
CLI's ``_log_and_report()``.

No auth, no CORS, no rate limiting in v1 (D14): this is an experimentation
gateway. The only filesystem a request can touch is the optional
``session_root`` (D9, enforced in deps).
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Tuple, Union

from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse

from ..cli import build_messages  # D1: reuse the CLI's pure choreography helper
from ..core.errors import FormatError, SessionError
from ..core.logger import log_request
from ..core.orchestrator import run_turn
from ..core.session import load_session_messages, save_session_messages
from ..core.telemetry import Timer
from ..core.types import ToolSpec
from ..providers import BaseProvider, ChatMessage, get_provider_spec, provider_names
from ..structured.extractor import extract as run_extraction
from ..token_utils import count_message_tokens, count_method, count_tokens
from ..tools.executor import ToolExecutor
from ..tools.registry import get_tools as get_registered_tools, tool_names
from . import deps
from .error_handlers import register_handlers
from .mappers import (
    extraction_result_to_structured_response,
    messages_in_to_chat_messages,
    orchestration_result_to_chat_response,
)
from .schemas import (
    ChatRequest,
    ChatResponse,
    ChatStreamRequest,
    StructuredRequest,
    StructuredResponse,
)
from .streaming import sse_chat_stream


def create_app(
    session_root: Optional[Union[str, Path]] = None,
    log_path: Optional[Union[str, Path]] = None,
) -> FastAPI:
    """Build the FastAPI app: routes, error handlers, minimal lifespan.

    ``session_root`` (D9): when given, ``session_path`` requests are
    confined under it; when ``None``, sessions over HTTP are disabled and
    any ``session_path`` request is rejected 400 before any provider call.

    ``log_path`` (D12): when given, every JSONL record this app writes —
    success records here, error records in error_handlers — goes to that
    file instead of the logger's env-coupled default, without touching
    ``LLM_GATEWAY_LOG_PATH`` semantics.
    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Intentionally minimal: no startup side effects. Provider
        # construction is per-request by design (D10), both registries are
        # populated at import time, and the logger opens its file per
        # record. Nothing here to initialize or tear down (yet).
        yield

    app = FastAPI(title="LLM Gateway", version="0.1.0", lifespan=lifespan)
    register_handlers(app, log_path=log_path)

    def _log_kwargs() -> dict:
        # Same None-omission pattern as error_handlers.log_gateway_error:
        # log_path=None means the logger's own (env-coupled) default.
        return {"log_path": log_path} if log_path is not None else {}

    # ------------------------------------------------------------------
    # Per-request resolution (Step 4's framework-agnostic deps, bound as
    # Depends here). The order below is the CLI's own choreography —
    # schema, then tools, then session, then provider construction — so
    # every request-shaped failure surfaces before a provider exists.
    # ------------------------------------------------------------------

    def _resolve_chat_schema(body: ChatRequest) -> Optional[dict]:
        return deps.validate_schema(body.schema_) if body.schema_ is not None else None

    def _resolve_chat_tools(
        body: ChatRequest,
    ) -> Tuple[Optional[List[ToolSpec]], Optional[ToolExecutor]]:
        if not body.tools:
            return None, None
        return deps.resolve_tools(body.tools)

    def _resolve_chat_session(body: ChatRequest) -> Optional[str]:
        return deps.resolve_session_path(body.session_path, session_root)

    def _resolve_chat_provider(body: ChatRequest) -> BaseProvider:
        return deps.resolve_provider(body.provider, body.model, timeout=body.timeout)

    def _resolve_structured_schema(body: StructuredRequest) -> dict:
        return deps.validate_schema(body.schema_)

    def _resolve_structured_provider(body: StructuredRequest) -> BaseProvider:
        return deps.resolve_provider(body.provider, body.model, timeout=body.timeout)

    def _load_session_and_assemble(
        body: ChatRequest, *, schema: Optional[dict], session_path: Optional[str]
    ) -> Tuple[List[ChatMessage], List[str]]:
        """Session load + guard + message assembly in one pass — the CLI's
        exact choreography (D1), shared by /v1/chat and /v1/chat/stream.

        Load errors (`SessionError`: the path exists but holds no valid
        session records) are re-raised as request-shaped `FormatError`s so
        the Step 3 handler turns them into a 400 *before any provider call*
        — mirroring the CLI, where a corrupt session file is reported before
        the turn runs.

        A continuation (prior history exists) does NOT re-inject
        system/schema: the saved transcript already carries them — and the
        CLI's non-fatal stderr notes for those cases surface here as
        entries in the response's `warnings` list (Step 7's decided
        channel), HTTP still 200. A first turn (no file yet) notes nothing.
        """
        warnings: List[str] = []
        prior_messages = None
        if session_path:
            try:
                prior_messages = load_session_messages(session_path)  # None when absent
            except SessionError as exc:
                raise FormatError(f"Unusable session file {session_path}: {exc}") from exc
            if prior_messages:
                if body.system:
                    warnings.append(
                        "`system` is ignored when continuing an existing session; "
                        "the session's saved system message takes precedence."
                    )
                if body.schema_ is not None:
                    warnings.append(
                        "`schema` is not re-injected when continuing an existing session; "
                        "the session's saved schema instruction takes precedence."
                    )
        if body.messages is not None:
            turn_input = messages_in_to_chat_messages(body.messages)
        else:
            turn_input = build_messages(
                None if prior_messages is not None else body.system,
                body.prompt,  # type: ignore[arg-type]  # validator guarantees one form
                schema=schema if prior_messages is None else None,
            )
        return list(prior_messages or []) + turn_input, warnings

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/healthz")
    def healthz() -> dict:
        """Liveness only: no provider call, no registry read, no log record."""
        return {"status": "ok"}

    @app.get("/v1/providers")
    def list_providers() -> dict:
        """Registered providers + default models — read from the registry
        itself (the same source the CLI's --provider choices come from)."""
        return {
            "providers": [
                {"name": name, "default_model": get_provider_spec(name).default_model}
                for name in provider_names()
            ]
        }

    @app.get("/v1/tools")
    def list_tools() -> dict:
        """Registered tools + their JSON-Schema parameters (D13's discovery
        surface: clients can see which names /v1/chat `tools` accepts)."""
        names = tool_names()
        return {
            "tools": [
                {
                    "name": registered.spec.name,
                    "description": registered.spec.description,
                    "parameters": registered.spec.parameters,
                }
                for registered in get_registered_tools(names)
            ]
        }

    @app.post("/v1/chat", response_model=ChatResponse)
    def chat(
        body: ChatRequest,
        schema: Optional[dict] = Depends(_resolve_chat_schema),
        tool_bundle: Tuple[Optional[List[ToolSpec]], Optional[ToolExecutor]] = Depends(
            _resolve_chat_tools
        ),
        session_path: Optional[str] = Depends(_resolve_chat_session),
        provider: BaseProvider = Depends(_resolve_chat_provider),
    ) -> ChatResponse:
        if body.stream:
            # D8's loud-failure rule, applied at the routing level: this is
            # the non-streaming endpoint, and answering a stream=true request
            # with a full JSON body would silently downgrade the contract.
            # /v1/chat/stream (Step 6) is the streaming surface.
            raise FormatError(
                "stream=true is not supported on POST /v1/chat; use POST /v1/chat/stream."
            )
        tool_specs, tool_executor = tool_bundle
        timer = Timer().start()

        # Step 7's session choreography: load errors → 400 pre-provider,
        # continuation notes as response `warnings`.
        messages, warnings = _load_session_and_assemble(body, schema=schema, session_path=session_path)

        # Client-side pre-count — identical to the CLI: the pre-request
        # context signal and the tokens_in fallback when the provider
        # doesn't bill/report usage.
        pre_count_tokens_in = count_message_tokens([m.to_content_dict() for m in messages])

        result = run_turn(
            provider,
            messages,
            tools=tool_specs,
            tool_executor=tool_executor,
            response_schema=schema,
            max_tool_iterations=body.max_tool_iterations,
            max_retries=body.max_retries,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        )

        # Session save — after success only; a failed turn never persists
        # (contract #5). A save failure must not fail a completed turn, so
        # it degrades to a warning on the response (Step 7's stderr
        # analogue) appended to any continuation notes from the guard.
        if session_path:
            try:
                save_session_messages(session_path, result.messages)
            except Exception as exc:  # noqa: BLE001  # the answer already succeeded
                warnings.append(f"Could not update session file {session_path}: {exc}")
        timer.stop()

        response = orchestration_result_to_chat_response(
            result,
            provider=provider.name,
            model=provider.model,
            pre_count_tokens_in=pre_count_tokens_in,
            warnings=warnings,
        )

        # Success log — CLI-identical fields and preference rules (D12):
        # provider-billed tokens_in wins; else the pre-count + method label.
        logged_tokens_in = result.tokens_in if result.tokens_in is not None else pre_count_tokens_in
        token_count_method_label = None if result.tokens_in is not None else count_method()
        log_request(
            provider=provider.name,
            latency_ms=timer.elapsed_ms,
            tokens_in=logged_tokens_in,
            tokens_out=result.tokens_out,
            temperature=body.temperature,
            status="success",
            error_type=None,
            token_count_method=token_count_method_label,
            tool_calls=result.tool_call_count,
            tool_iterations=result.tool_iterations,
            **_log_kwargs(),
        )
        return response

    @app.post("/v1/chat/stream")
    def chat_stream(
        body: ChatStreamRequest,
        session_path: Optional[str] = Depends(_resolve_chat_session),
        provider: BaseProvider = Depends(_resolve_chat_provider),
    ) -> StreamingResponse:
        """SSE streaming chat — plain chat only (D8).

        Tools/schema never reach this route: `ChatStreamRequest` pins the
        inherited D8 validator, so those requests are 422 at body
        validation, before the response starts. All other request-shaped
        failures (session/provider deps) raise here too — normal statuses,
        no SSE headers sent. Once the stream starts, every provider error
        becomes an in-band error event (streaming.py).
        """
        timer = Timer().start()
        # Step 7's session choreography (same as /v1/chat: 400 on load
        # errors, continuation notes in the done event's `warnings`).
        messages, warnings = _load_session_and_assemble(body, schema=None, session_path=session_path)
        pre_count_tokens_in = count_message_tokens([m.to_content_dict() for m in messages])

        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # nginx: don't buffer the event stream
        }
        return StreamingResponse(
            sse_chat_stream(
                provider,
                messages,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
                pre_count_tokens_in=pre_count_tokens_in,
                timer=timer,
                session_path=session_path,
                log_path=log_path,
                warnings=warnings,
            ),
            media_type="text/event-stream",
            headers=headers,
        )

    @app.post("/v1/structured", response_model=StructuredResponse)
    def structured(
        body: StructuredRequest,
        schema: dict = Depends(_resolve_structured_schema),
        provider: BaseProvider = Depends(_resolve_structured_provider),
    ) -> StructuredResponse:
        timer = Timer().start()
        # Pre-count identical to the CLI's structured command: input text
        # plus the serialized schema (both go to the provider).
        pre_count_tokens_in = count_tokens(body.text) + count_tokens(json.dumps(schema))

        result = run_extraction(
            provider,
            body.text,
            schema,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
            max_retries=body.max_retries,
        )
        timer.stop()

        response = extraction_result_to_structured_response(
            result,
            provider=provider.name,
            model=provider.model,
            pre_count_tokens_in=pre_count_tokens_in,
        )

        logged_tokens_in = result.tokens_in if result.tokens_in is not None else pre_count_tokens_in
        token_count_method_label = None if result.tokens_in is not None else count_method()
        log_request(
            provider=provider.name,
            latency_ms=timer.elapsed_ms,
            tokens_in=logged_tokens_in,
            tokens_out=result.tokens_out,
            temperature=body.temperature,
            status="success",
            error_type=None,
            token_count_method=token_count_method_label,
            **_log_kwargs(),
        )
        return response

    return app


# Module-level instance (D3): makes `uvicorn src.api.app:app` work with no
# flags. Constructed with no session_root (sessions disabled by default)
# and the logger's env-coupled default log path.
app = create_app()
