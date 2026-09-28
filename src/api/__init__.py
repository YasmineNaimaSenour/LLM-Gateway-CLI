"""HTTP front-end over the provider-agnostic runtime — the second thin shell.

Peer of :mod:`src.cli`: message assembly, HTTP I/O, and error choreography only.
All capabilities (orchestration, providers, structured extraction, tool calling,
sessions, logging) are consumed from the existing runtime packages; nothing is
re-implemented here.

Note: this package exists for the HTTP API layer only. No module outside
``src/api/`` imports FastAPI/uvicorn, keeping the runtime's dependency
footprint unchanged (API_LAYER_PLAN.md, D2).

Step 0 (API_LAYER_PLAN.md §7): package created intentionally empty — this module
exports nothing yet. ``create_app`` is introduced with ``src/api/app.py`` in
Step 5; ``__all__`` will then declare it (D15).
"""
