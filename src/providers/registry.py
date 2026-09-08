"""Provider registry: how the gateway learns about backends without the CLI
naming a single concrete provider class.

Once the defining module is imported, the provider is first-class: the CLI's
`--provider` choices, `--model` defaulting, and instantiation all flow from
this registry, and the CLI needs zero edits.

The registry holds `ProviderSpec` records, not instances: instantiation is
deferred to `get_provider()` so that constructors stay cheap and errorful
side effects (e.g. GroqProvider raising ModelError when GROQ_API_KEY is
missing) happen at call time, where the CLI's error handling lives — not at
import time of an unrelated module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Type

from ..core.errors import FormatError
from .base import BaseProvider


@dataclass(frozen=True)
class ProviderSpec:
    """A provider as known to the runtime: the class that implements it and
    the model to use when the caller doesn't name one."""

    name: str
    cls: Type[BaseProvider]
    default_model: Optional[str] = None


_REGISTRY: Dict[str, ProviderSpec] = {}


def register_provider(
    name: str, *, default_model: Optional[str] = None
) -> Callable[[Type[BaseProvider]], Type[BaseProvider]]:
    """Class decorator: make a BaseProvider subclass available to the CLI."""

    def decorator(cls: Type[BaseProvider]) -> Type[BaseProvider]:
        if name in _REGISTRY:
            raise FormatError(f"Provider {name!r} is already registered.")
        _REGISTRY[name] = ProviderSpec(name=name, cls=cls, default_model=default_model)
        return cls

    return decorator


def get_provider_spec(name: str) -> ProviderSpec:
    """Look up a provider by name. Raises FormatError for unknown names.

    The CLI validates `--provider` before ever touching a network — this is
    a caller/config check, distinct from a provider failing at call time
    (rate limit, timeout, ...), which the error taxonomy handles instead.
    """
    spec = _REGISTRY.get(name)
    if spec is None:
        available = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise FormatError(f"Unknown provider: {name!r}. Available providers: {available}.")
    return spec


def get_provider(name: str, model: Optional[str] = None) -> BaseProvider:
    """Instantiate a provider, falling back to its registered default model.

    Raises FormatError for an unregistered name. Construction errors (missing
    API keys, etc.) propagate from the provider's own __init__ and are the
    provider's business, not the registry's.
    """
    spec = get_provider_spec(name)
    return spec.cls(model=model or spec.default_model)


def provider_names() -> List[str]:
    """All registered provider names (sorted)."""
    return sorted(_REGISTRY)


def unregister_provider(name: str) -> None:
    """Remove a provider registration. Exists for test isolation — the
    registry, like the tool registry, is intentionally module-level global
    state (audit #8 covers making that instance-based before server mode)."""
    _REGISTRY.pop(name, None)
