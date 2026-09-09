"""LLM Gateway: a provider-agnostic CLI/runtime for LLM experimentation.

Top-level entry points are the CLI (`python -m src.cli`) and the runtime
core (`src.core.orchestrator.run_turn`). This package deliberately exports
nothing itself: the public API surface is the union of the subpackages'
declared `__all__` lists (src.core, src.providers, src.structured,
src.tools). Anything not named there is an implementation detail and may
change without notice.
"""

# Declared surface: empty at the top level by design (see docstring).
__all__: list[str] = []
