"""Provider registry: how the gateway learns about backends without the CLI
naming a single concrete provider class.

Once the defining module is imported, the provider is first-class: the CLI's
`--provider` choices, `--model` defaulting, and instantiation all flow from
this registry, and the CLI needs zero edits.

Like the tool registry, registry mechanics live in a class whose
state is instance-level; a single module-level `DEFAULT_REGISTRY` instance
preserves the import-time registration flow, and the module-level function
aliases keep every existing call site unchanged. The registry holds
`ProviderSpec` records, not instances: instantiation is deferred to
`get_provider()` so that constructors stay cheap and errorful side effects
(e.g. GroqProvider raising ModelError when GROQ_API_KEY is missing) happen
at call time, where the CLI's error handling lives — not at import time of
an unrelated module.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Type

from ..core.errors import FormatError
from .base import BaseProvider


def _init_params(cls: type) -> set:
    """Parameter names of a provider class's constructor.

    Used for capability detection ("does this provider accept a timeout?")
    without importing concrete provider classes into the registry.
    """
    try:
        return set(inspect.signature(cls.__init__).parameters)
    except (TypeError, ValueError):  # builtins/C-level constructors
        return set()


@dataclass(frozen=True)
class ProviderSpec:
    """A provider as known to the runtime: the class that implements it and
    the model to use when the caller doesn't name one."""

    name: str
    cls: Type[BaseProvider]
    default_model: Optional[str] = None


class ProviderRegistry:
    """An instance-isolated set of registered providers.

    One instance per runtime (the CLI uses `DEFAULT_REGISTRY`); tests can
    create private instances for isolation, and a future server mode can
    pass instances around explicitly instead of sharing process globals.
    """

    def __init__(self) -> None:
        self._providers: Dict[str, ProviderSpec] = {}

    def register_provider(
        self, name: str, *, default_model: Optional[str] = None
    ) -> Callable[[Type[BaseProvider]], Type[BaseProvider]]:
        """Class decorator: make a BaseProvider subclass available to the CLI."""

        def decorator(cls: Type[BaseProvider]) -> Type[BaseProvider]:
            if name in self._providers:
                raise FormatError(f"Provider {name!r} is already registered.")
            self._providers[name] = ProviderSpec(name=name, cls=cls, default_model=default_model)
            return cls

        return decorator

    def get_provider_spec(self, name: str) -> ProviderSpec:
        """Look up a provider by name. Raises FormatError for unknown names.

        The CLI validates `--provider` before ever touching a network — this
        is a caller/config check, distinct from a provider failing at call
        time (rate limit, timeout, ...), which the error taxonomy handles
        instead.
        """
        spec = self._providers.get(name)
        if spec is None:
            available = ", ".join(sorted(self._providers)) or "(none registered)"
            raise FormatError(f"Unknown provider: {name!r}. Available providers: {available}.")
        return spec

    def get_provider(
        self, name: str, model: Optional[str] = None, *, timeout: Optional[float] = None
    ) -> BaseProvider:
        """Instantiate a provider, falling back to its registered default model.

        `timeout`, when given, is forwarded to providers whose constructor
        accepts a request-timeout knob (detected by signature, so the
        registry stays provider-agnostic — no concrete class imports here;
        both built-ins currently accept one). Raises FormatError
        for an unregistered name. Construction errors (missing API keys,
        etc.) propagate from the provider's own __init__ and are the
        provider's business, not the registry's.
        """
        spec = self.get_provider_spec(name)
        kwargs: Dict[str, object] = {}
        if timeout is not None and "timeout" in _init_params(spec.cls):
            kwargs["timeout"] = timeout
        return spec.cls(model=model or spec.default_model, **kwargs)

    def provider_names(self) -> List[str]:
        """All registered provider names (sorted)."""
        return sorted(self._providers)

    def unregister_provider(self, name: str) -> None:
        """Remove a provider registration. Exists for test isolation."""
        self._providers.pop(name, None)

    def snapshot(self) -> Dict[str, ProviderSpec]:
        """A copy of the registration table — for test isolation helpers."""
        return dict(self._providers)

    def restore(self, snapshot: Dict[str, ProviderSpec]) -> None:
        """Replace the registration table with a previous snapshot."""
        self._providers = dict(snapshot)


# The module-level default instance preserves the existing import-time
# registration flow; the bound-function aliases below keep every existing
# call site (`@register_provider`, `get_provider(...)`, `provider_names()`)
# working unchanged.
DEFAULT_REGISTRY = ProviderRegistry()

register_provider = DEFAULT_REGISTRY.register_provider
get_provider = DEFAULT_REGISTRY.get_provider
get_provider_spec = DEFAULT_REGISTRY.get_provider_spec
provider_names = DEFAULT_REGISTRY.provider_names
unregister_provider = DEFAULT_REGISTRY.unregister_provider
