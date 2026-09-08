"""Provider package: BaseProvider implementations plus the registry that
makes them discoverable.

Importing this package registers the built-in providers — the
@register_provider decorator in each provider module runs at import time.
Adding a provider is then a one-file, zero-CLI-edit operation: implement
BaseProvider, decorate it with @register_provider, and make sure the module
is imported once (by listing it below, or by the consumer importing it).

The public API of this package is the registry; concrete provider classes
are reachable as attributes of their own modules, not re-exported here.
"""

from .base import BaseProvider, ChatMessage, ChatResponse
from .registry import (
    ProviderSpec,
    get_provider,
    get_provider_spec,
    provider_names,
    register_provider,
    unregister_provider,
)

# Side-effect imports: run each built-in provider's @register_provider
# decorator. Nothing else in the package needs these names, hence noqa.
from . import ollama_provider as _ollama_provider  # noqa: F401
from . import groq_provider as _groq_provider  # noqa: F401

__all__ = [
    "BaseProvider",
    "ChatMessage",
    "ChatResponse",
    "ProviderSpec",
    "get_provider",
    "get_provider_spec",
    "provider_names",
    "register_provider",
    "unregister_provider",
]
