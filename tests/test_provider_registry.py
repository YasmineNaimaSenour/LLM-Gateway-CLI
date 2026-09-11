"""Tests for the provider registry: the mechanism that lets a
new backend become first-class — argparse choices, model defaulting,
instantiation — with zero CLI edits, provided its module gets imported.
"""

from __future__ import annotations

import pytest

from src.cli import build_parser, build_provider
from src.core.errors import FormatError
from src.providers import (
    BaseProvider,
    ChatMessage,
    ChatResponse,
    get_provider,
    get_provider_spec,
    provider_names,
    register_provider,
    unregister_provider,
)
from src.providers.groq_provider import GroqProvider
from src.providers.ollama_provider import DEFAULT_OLLAMA_TIMEOUT, OllamaProvider


class _DummyProvider(BaseProvider):
    name = "dummy"

    def chat(self, messages, **kwargs):
        return ChatResponse(text="dummy", tokens_out=1)

    def chat_stream(self, messages, **kwargs):
        yield "dummy"


@pytest.fixture
def clean_registry():
    """Snapshot the registry around each test so dummy registrations and
    unregistrations never leak into other tests (uses the registry's own
    snapshot/restore helpers rather than its private state)."""
    from src.providers import registry

    snapshot = registry.DEFAULT_REGISTRY.snapshot()
    yield
    registry.DEFAULT_REGISTRY.restore(snapshot)


# ---------------------------------------------------------------------------
# built-in registration
# ---------------------------------------------------------------------------


def test_builtin_providers_are_registered_on_import():
    assert provider_names() == ["groq", "ollama"]


def test_get_provider_returns_built_in_with_its_default_model():
    provider = get_provider("ollama")
    assert isinstance(provider, OllamaProvider)
    assert provider.model == "llama3.2"


def test_get_provider_explicit_model_overrides_the_default():
    provider = get_provider("groq", model="llama-3.1-8b-instant")
    assert isinstance(provider, GroqProvider)
    assert provider.model == "llama-3.1-8b-instant"


def test_unknown_provider_raises_format_error_listing_available():
    with pytest.raises(FormatError) as excinfo:
        get_provider("does_not_exist")
    message = str(excinfo.value)
    assert "does_not_exist" in message
    assert "ollama" in message and "groq" in message


# ---------------------------------------------------------------------------
# registration lifecycle
# ---------------------------------------------------------------------------


def test_register_provider_makes_a_new_provider_instantiable(clean_registry):
    register_provider("dummy", default_model="dummy-model")(_DummyProvider)

    provider = get_provider("dummy")
    assert isinstance(provider, _DummyProvider)
    assert provider.model == "dummy-model"
    assert "dummy" in provider_names()


def test_duplicate_provider_name_is_rejected(clean_registry):
    register_provider("dummy")(_DummyProvider)
    with pytest.raises(FormatError, match="already registered"):
        register_provider("dummy")(_DummyProvider)


def test_unregister_provider_removes_it(clean_registry):
    register_provider("dummy")(_DummyProvider)
    unregister_provider("dummy")

    assert "dummy" not in provider_names()
    with pytest.raises(FormatError, match="Unknown provider"):
        get_provider("dummy")


def test_unregister_unknown_name_is_a_no_op(clean_registry):
    unregister_provider("never_registered")  # must not raise


def test_get_provider_spec_carries_default_model():
    spec = get_provider_spec("ollama")
    assert spec.name == "ollama"
    assert spec.cls is OllamaProvider
    assert spec.default_model == "llama3.2"


# ---------------------------------------------------------------------------
# timeout forwarding
# ---------------------------------------------------------------------------


def test_get_provider_forwards_timeout_to_providers_with_a_timeout_knob():
    # --timeout flows CLI -> build_provider -> registry -> constructor; the
    # registry detects the knob by constructor signature, so it stays
    # provider-agnostic (no concrete class imports).
    provider = get_provider("ollama", None, timeout=33.0)
    assert provider.timeout == 33.0
    provider = get_provider("groq", None, timeout=21.0)
    assert provider.timeout == 21.0


def test_get_provider_without_timeout_leaves_provider_defaults_intact():
    # No --timeout flag: the provider's own env/default precedence applies
    # (OLLAMA_TIMEOUT, GROQ_TIMEOUT, 60s defaults) — the registry adds nothing.
    provider = get_provider("ollama", None)
    assert provider.timeout == DEFAULT_OLLAMA_TIMEOUT


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_cli_build_provider_delegates_to_the_registry():
    provider = build_provider("ollama", None)
    assert isinstance(provider, OllamaProvider)
    assert provider.model == "llama3.2"


def test_cli_provider_choices_come_from_the_registry(clean_registry):
    register_provider("dummy")(_DummyProvider)

    parser = build_parser()
    args = parser.parse_args(["chat", "--provider", "dummy", "--prompt", "hi"])
    assert args.provider == "dummy"


def test_cli_rejects_provider_that_is_not_registered():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["chat", "--provider", "does_not_exist", "--prompt", "hi"])
