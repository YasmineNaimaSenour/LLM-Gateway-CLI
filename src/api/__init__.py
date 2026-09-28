"""HTTP front-end over the provider-agnostic runtime — the second thin shell.

Peer of :mod:`src.cli`: message assembly, HTTP I/O, and error choreography only.
All capabilities (orchestration, providers, structured extraction, tool calling,
sessions, logging) are consumed from the existing runtime packages; nothing is
re-implemented here.

Note: this package exists for the HTTP API layer only. No module outside
``src/api/`` imports FastAPI/uvicorn, keeping the runtime's dependency
footprint unchanged (API_LAYER_PLAN.md, D2).

Public API (D15): ``create_app`` — the only object other modules should
import. The routes, dependencies, mappers, and error handlers are the
package's internals.
"""

from .app import create_app

__all__ = ["create_app"]
