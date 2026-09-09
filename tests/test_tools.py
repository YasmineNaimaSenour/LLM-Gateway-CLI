import pytest

from src.core.errors import FormatError
from src.core.types import ToolCall
from src.tools.executor import ToolExecutor
from src.tools.registry import ToolRegistry, get_tools


def test_get_tools_resolves_registered_names():
    tools = get_tools(["calculator", "current_time"])
    assert {t.spec.name for t in tools} == {"calculator", "current_time"}
    assert tools[0].spec.parameters["type"] == "object"


def test_get_tools_raises_format_error_for_unknown_name():
    with pytest.raises(FormatError):
        get_tools(["not_a_real_tool"])


def test_private_registry_instances_are_isolated_from_the_default():
    # audit #8: registry state is instance-level, so tests (or a future
    # server mode) can hold independent registries without touching the
    # module-level default that the built-in tools register on.
    private = ToolRegistry()

    @private.register(
        "echo",
        "Echo the given message back.",
        {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
    )
    def _echo(message: str) -> str:
        return message

    assert private.names() == ["echo"]
    assert "echo" not in get_tools.__self__.names()  # default registry untouched
    with pytest.raises(FormatError):
        private.get_tools(["calculator"])  # built-ins live on the default registry


def test_executor_runs_calculator_successfully():
    executor = ToolExecutor(get_tools(["calculator"]))
    result = executor.execute(ToolCall(id="1", name="calculator", arguments={"a": 2, "b": 3, "operation": "add"}))
    assert result.is_error is False
    assert result.content == "5.0"
    assert result.tool_call_id == "1"


def test_executor_returns_error_result_for_unknown_tool_without_raising():
    executor = ToolExecutor(get_tools(["calculator"]))
    result = executor.execute(ToolCall(id="1", name="ghost_tool", arguments={}))
    assert result.is_error is True
    assert "No such tool" in result.content


def test_executor_returns_error_result_for_invalid_arguments():
    executor = ToolExecutor(get_tools(["calculator"]))
    result = executor.execute(ToolCall(id="1", name="calculator", arguments={"a": "not-a-number", "b": 1, "operation": "add"}))
    assert result.is_error is True
    assert "Invalid arguments" in result.content


def test_executor_returns_error_result_when_tool_function_raises():
    executor = ToolExecutor(get_tools(["calculator"]))
    result = executor.execute(ToolCall(id="1", name="calculator", arguments={"a": 1, "b": 0, "operation": "divide"}))
    assert result.is_error is True
    assert "Division by zero" in result.content


def test_executor_serializes_non_string_results_as_json():
    executor = ToolExecutor(get_tools(["calculator"]))
    result = executor.execute(ToolCall(id="1", name="calculator", arguments={"a": 4, "b": 2, "operation": "multiply"}))
    assert result.content == "8.0"