"""Per-request resolution dependencies (§6): the API layer's single seam.

Each dependency turns one piece of request vocabulary into the runtime
object the route handler needs, converting request-shaped errors into
`FormatError`s **before any provider call** — mirroring the CLI's
startup-vs-calltime distinction:

  - `resolve_provider`     — registry lookup + instantiation (D10): per
    request, never cached, so constructor side effects (e.g. Groq without
    GROQ_API_KEY → ModelError) happen at call time, inside error handling.
  - `resolve_tools`        — names → RegisteredTools + ToolExecutor (D13).
  - `validate_schema`      — re-export of the Step 2 inline-schema helper.
  - `resolve_session_path` — session_root confinement (D9): the server's
    opt-in root is the only filesystem a request may touch.

The module-level function forms take the registry/executor objects
explicitly (pure functions, trivially unit-testable); the FastAPI
`Depends` wrappers in app.py (Step 5) bind them to request parameters.
No FastAPI imports here — deps.py stays framework-agnostic like the rest
of the API layer's seams.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from ..core.errors import FormatError
from ..core.types import ToolSpec
from ..providers.base import BaseProvider
from ..providers.registry import get_provider as _registry_get_provider
from ..tools.executor import ToolExecutor
from ..tools.registry import get_tools as _registry_get_tools
from .mappers import validate_inline_schema

__all__ = [
    "resolve_provider",
    "resolve_tools",
    "validate_schema",
    "resolve_session_path",
    "validate_session_path",
]


# ---------------------------------------------------------------------------
# Provider resolution (D10)
# ---------------------------------------------------------------------------


def resolve_provider(
    name: str,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
) -> BaseProvider:
    """Instantiate the named provider via the registry — per request (D10).

    `model=None` → the registry's per-provider default; `timeout` is
    forwarded registry-style (only accepted by providers with the knob).
    Unknown names raise the registry's own `FormatError` ("Available
    providers: ...") — mapped to 400 by the Step 3 handler. Constructor
    errors (e.g. Groq without GROQ_API_KEY → ModelError) propagate
    unchanged: they are call-time model failures (→ 502), not request
    validation failures.
    """
    return _registry_get_provider(name, model, timeout=timeout)


# ---------------------------------------------------------------------------
# Tool resolution (D13)
# ---------------------------------------------------------------------------


def resolve_tools(names: List[str]) -> Tuple[List[ToolSpec], ToolExecutor]:
    """Resolve tool names → (wire specs, executor), pre-network.

    Unknown names raise `FormatError` (→ 400) with the runtime's own
    "Available tools: ..." message — the runtime's startup-time check,
    kept exactly where the CLI has it. The executor's never-raises
    contract (REPORT §7.8) means tool-domain failures stay inside the
    conversation after this point; the API layer never sees them.
    """
    registered = _registry_get_tools(list(names))
    return [t.spec for t in registered], ToolExecutor(registered)


# ---------------------------------------------------------------------------
# Inline schema (D7)
# ---------------------------------------------------------------------------


def validate_schema(schema) -> dict:
    """Validate a request's inline schema (re-export of the Step 2 helper).

    `SchemaError` → 424, `UnsupportedSchemaError` → 422 via the Step 3
    handler; both fire before any provider call.
    """
    return validate_inline_schema(schema)


# ---------------------------------------------------------------------------
# Session path confinement (D9)
# ---------------------------------------------------------------------------


def validate_session_path(session_path: str, session_root: Optional[str]) -> str:
    """Confine a requested session path to the app's `session_root` (D9).

    Rules (each rejected with `FormatError` → 400, pre-provider):
      - feature disabled (`session_root=None`) and a path was requested;
      - absolute path — clients never name absolute locations;
      - path escaping the root via `..` or symlink-style traversal
        (checked textually AND on the resolved path).

    Returns the resolved path (root / relative-request) for the session
    helpers to use. Raises `FormatError` — not `SessionError` — on
    confinement failures: the request references a location the server
    does not permit, which is a client error, not a corrupt-file problem.
    """
    if session_root is None:
        raise FormatError(
            "Sessions over HTTP are disabled on this server (no session_root configured); "
            "use `messages` in/out instead."
        )

    root = Path(session_root).expanduser()
    requested = Path(session_path).expanduser()

    if requested.is_absolute():
        raise FormatError(
            f"session_path must be relative to the server's session root, got absolute path: {session_path!r}."
        )

    candidate = root / requested

    # Textual check: catch ..-traversal regardless of the filesystem.
    if ".." in requested.parts:
        raise FormatError(f"session_path must stay inside the session root: {session_path!r} escapes it.")

    # Resolved check: belt-and-suspenders (root itself being relative,
    # exotic platform behavior); never raises for a non-existent leaf,
    # which is legitimate — first turn creates the file.
    resolved_root = root.resolve()
    resolved_candidate = candidate.resolve()
    if resolved_candidate != resolved_root and resolved_root not in resolved_candidate.parents:
        raise FormatError(f"session_path must stay inside the session root: {session_path!r} escapes it.")

    return str(candidate)


def resolve_session_path(session_path: Optional[str], session_root: Optional[str]) -> Optional[str]:
    """None-tolerant wrapper for route handlers: None in → None out."""
    if session_path is None:
        return None
    return validate_session_path(session_path, session_root)
