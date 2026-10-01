"""Tests for src/api/deps.py — the API layer's per-request resolution seam.

Unit tests for the API layer's resolution seam — registry-driven, no
FastAPI app, no network. Registry isolation uses the runtime's own
snapshot()/restore() mechanism; provider-constructor errors are exercised
with a fake provider registered on DEFAULT_REGISTRY (no concrete provider
imports, no env coupling).
"""

from typing import Optional

import pytest

from src.api import deps
from src.core.errors import FormatError, ModelError
from src.core.types import ToolCall, ToolSpec
from src.providers.base import BaseProvider
from src.providers.registry import DEFAULT_REGISTRY as PROVIDER_REGISTRY, ProviderSpec
from src.tools.executor import ToolExecutor
from src.tools.registry import DEFAULT_REGISTRY as TOOL_REGISTRY

VALID_SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
}


# ---------------------------------------------------------------------------
# Registry isolation
# ---------------------------------------------------------------------------


@pytest.fixture()
def provider_registry_snapshot():
    snap = PROVIDER_REGISTRY.snapshot()
    yield PROVIDER_REGISTRY
    PROVIDER_REGISTRY.restore(snap)


@pytest.fixture()
def tool_registry_snapshot():
    snap = TOOL_REGISTRY.snapshot()
    yield TOOL_REGISTRY
    TOOL_REGISTRY.restore(snap)


class _FakeProvider(BaseProvider):
    name = "fake"

    def __init__(self, model: str = "fake-mini", timeout: Optional[float] = None):  # noqa: F821
        self.model = model
        self.timeout = 30.0 if timeout is None else float(timeout)

    def chat(self, messages, *, temperature=0.7, max_tokens=512, tools=None, response_schema=None):
        raise NotImplementedError

    def chat_stream(self, messages, *, temperature=0.7, max_tokens=512):
        yield from ()


class _ExplodingProvider(BaseProvider):
    """Constructor side effect (D10's payoff): fails at call time, like
    GroqProvider without GROQ_API_KEY — without env coupling."""

    name = "exploding"

    def __init__(self, model: str = "boom-1"):
        raise ModelError("Missing credential for exploding provider.", provider=self.name)

    def chat(self, messages, **kwargs):
        raise NotImplementedError

    def chat_stream(self, messages, **kwargs):
        yield from ()


# ---------------------------------------------------------------------------
# Provider resolution (D10)
# ---------------------------------------------------------------------------


class TestResolveProvider:
    def test_known_provider_resolves_with_default_model(self, provider_registry_snapshot):
        PROVIDER_REGISTRY._providers["fake"] = ProviderSpec(name="fake", cls=_FakeProvider, default_model="fake-mini")
        provider = deps.resolve_provider("fake")
        assert isinstance(provider, _FakeProvider)
        assert provider.model == "fake-mini"  # None → spec.default_model

    def test_explicit_model_overrides_the_default(self, provider_registry_snapshot):
        PROVIDER_REGISTRY._providers["fake"] = ProviderSpec(name="fake", cls=_FakeProvider, default_model="fake-mini")
        provider = deps.resolve_provider("fake", "fake-pro")
        assert provider.model == "fake-pro"

    def test_unknown_provider_raises_the_registry_format_error_listing_available(self, provider_registry_snapshot):
        with pytest.raises(FormatError) as excinfo:
            deps.resolve_provider("nope")
        assert "Available providers" in str(excinfo.value)

    def test_timeout_is_forwarded_to_the_constructed_provider(self, provider_registry_snapshot):
        PROVIDER_REGISTRY._providers["fake"] = ProviderSpec(name="fake", cls=_FakeProvider, default_model="fake-mini")
        provider = deps.resolve_provider("fake", timeout=12.5)
        assert provider.timeout == 12.5  # constructed provider's .timeout

    def test_timeout_none_leaves_provider_default(self, provider_registry_snapshot):
        PROVIDER_REGISTRY._providers["fake"] = ProviderSpec(name="fake", cls=_FakeProvider, default_model="fake-mini")
        assert deps.resolve_provider("fake").timeout == 30.0

    def test_provider_constructor_failure_is_a_model_error_not_format_error(self, provider_registry_snapshot):
        # D10's payoff: construction happens at call time inside error
        # handling — the Step 3 handler maps this ModelError to 502, and it
        # must NOT be mistaken for a 400-shaped request problem.
        PROVIDER_REGISTRY._providers["exploding"] = ProviderSpec(
            name="exploding", cls=_ExplodingProvider, default_model="boom-1"
        )
        with pytest.raises(ModelError) as excinfo:
            deps.resolve_provider("exploding")
        assert "Missing credential" in str(excinfo.value)

    def test_real_groq_without_api_key_raises_model_error(self, provider_registry_snapshot, monkeypatch):
        # Concrete real-provider scenario, env-cleared (groq IS registered).
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(ModelError):
            deps.resolve_provider("groq")


# ---------------------------------------------------------------------------
# Tool resolution (D13)
# ---------------------------------------------------------------------------


class TestResolveTools:
    def test_known_names_resolve_to_specs_plus_executor(self):
        specs, executor = deps.resolve_tools(["calculator", "current_time"])
        assert [s.name for s in specs] == ["calculator", "current_time"]
        assert all(isinstance(s, ToolSpec) for s in specs)
        assert isinstance(executor, ToolExecutor)

    def test_unknown_tool_name_raises_format_error_listing_available(self):
        with pytest.raises(FormatError) as excinfo:
            deps.resolve_tools(["calculator", "no_such_tool"])
        assert "Unknown tool(s)" in str(excinfo.value)
        assert "Available tools" in str(excinfo.value)

    def test_unknown_tool_resolution_happens_before_any_provider_call(self):
        # The whole point of the seam: this raises before a provider exists.
        with pytest.raises(FormatError):
            deps.resolve_tools(["typo"])

    def test_resolved_executor_actually_executes_a_call(self):
        # D13 end-to-end: the executor handed to run_turn really runs the
        # calculator (3.0-style result), proving zero new tool machinery.
        _, executor = deps.resolve_tools(["calculator"])
        result = executor.execute(
            ToolCall(id="c1", name="calculator", arguments={"a": 1, "b": 2, "operation": "add"})
        )
        assert result.is_error is False
        assert result.content == "3.0"

    def test_valid_schema_helper_passes_through(self):
        schema = dict(VALID_SCHEMA)
        assert deps.validate_schema(schema) is schema

    def test_invalid_schema_raises_schema_error(self):
        from src.core.errors import SchemaError

        with pytest.raises(SchemaError):
            deps.validate_schema({"type": "object", "properties": {"a": {"type": "strin"}}, "required": ["a"]})


# ---------------------------------------------------------------------------
# Session path confinement (D9)
# ---------------------------------------------------------------------------


class TestSessionPathConfinement:
    def test_disabled_feature_rejects_any_request(self):
        with pytest.raises(FormatError) as excinfo:
            deps.validate_session_path("chats/demo.jsonl", session_root=None)
        assert "disabled" in str(excinfo.value)

    def test_relative_path_resolves_inside_the_root(self, tmp_path):
        resolved = deps.validate_session_path("chats/demo.jsonl", session_root=str(tmp_path))
        assert resolved == str(tmp_path / "chats" / "demo.jsonl")

    def test_parent_traversal_is_rejected(self, tmp_path):
        with pytest.raises(FormatError):
            deps.validate_session_path("../escape.jsonl", session_root=str(tmp_path))

    def test_sneaky_traversal_is_rejected(self, tmp_path):
        with pytest.raises(FormatError):
            deps.validate_session_path("chats/../../escape.jsonl", session_root=str(tmp_path))

    def test_absolute_path_is_rejected(self, tmp_path):
        with pytest.raises(FormatError):
            deps.validate_session_path(str(tmp_path / "abs.jsonl"), session_root=str(tmp_path))

    def test_root_boundary_itself_is_allowed(self, tmp_path):
        # A session file sitting at the root is inside the root.
        resolved = deps.validate_session_path("demo.jsonl", session_root=str(tmp_path))
        assert resolved == str(tmp_path / "demo.jsonl")

    def test_nested_new_subdirectories_are_permitted(self, tmp_path):
        resolved = deps.validate_session_path("a/b/c/session.jsonl", session_root=str(tmp_path))
        assert resolved == str(tmp_path / "a" / "b" / "c" / "session.jsonl")

    def test_resolve_session_path_none_is_none(self, tmp_path):
        assert deps.resolve_session_path(None, session_root=str(tmp_path)) is None
