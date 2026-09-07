from unittest.mock import MagicMock

import pytest

from src.core.errors import ToolLoopError
from src.core.orchestrator import run_turn
from src.core.types import ToolCall
from src.providers.base import ChatMessage, ChatResponse
from src.tools.executor import ToolExecutor
from src.tools.registry import get_tools

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
}


def _provider(*responses, name="fake"):
    provider = MagicMock()
    provider.name = name
    provider.chat.side_effect = list(responses)
    return provider


def _messages():
    return [ChatMessage(role="user", content="hi")]


def test_run_turn_without_tools_or_schema_makes_a_single_call():
    provider = _provider(ChatResponse(text="hello back", tokens_out=5))
    result = run_turn(provider, _messages())
    assert result.text == "hello back"
    assert result.tokens_out == 5
    assert result.tool_call_count == 0
    assert result.tool_iterations == 0
    assert provider.chat.call_count == 1


def test_run_turn_executes_tool_calls_then_returns_final_answer():
    call = ToolCall(id="1", name="calculator", arguments={"a": 1, "b": 2, "operation": "add"})
    provider = _provider(
        ChatResponse(text="", tokens_out=5, tool_calls=[call]),
        ChatResponse(text="The result is 3.", tokens_out=4, tool_calls=[]),
    )
    executor = ToolExecutor(get_tools(["calculator"]))
    messages = _messages()

    result = run_turn(provider, messages, tools=[get_tools(["calculator"])[0].spec], tool_executor=executor)

    assert result.text == "The result is 3."
    assert result.tool_call_count == 1
    assert result.tool_iterations == 1
    assert result.tokens_out == 9
    assert provider.chat.call_count == 2
    # an assistant (tool_calls) message and a tool (result) message were appended
    assert messages[-2].role == "assistant"
    assert messages[-1].role == "tool"
    assert messages[-1].content == "3.0"


def test_run_turn_raises_tool_loop_error_past_max_iterations():
    call = ToolCall(id="1", name="calculator", arguments={"a": 1, "b": 2, "operation": "add"})
    provider = _provider(*[ChatResponse(text="", tokens_out=1, tool_calls=[call]) for _ in range(3)])
    executor = ToolExecutor(get_tools(["calculator"]))

    with pytest.raises(ToolLoopError):
        run_turn(
            provider,
            _messages(),
            tools=[get_tools(["calculator"])[0].spec],
            tool_executor=executor,
            max_tool_iterations=3,
        )
    assert provider.chat.call_count == 3


def test_run_turn_requires_tool_executor_when_tools_given():
    provider = _provider(ChatResponse(text="hi", tokens_out=1))
    with pytest.raises(ValueError):
        run_turn(provider, _messages(), tools=[get_tools(["calculator"])[0].spec])


def test_run_turn_schema_only_delegates_to_coerce_to_schema():
    provider = _provider(
        ChatResponse(text="not json at all", tokens_out=3),
        ChatResponse(text='{"name": "Bob", "age": 30}', tokens_out=6),
    )
    result = run_turn(provider, _messages(), response_schema=SCHEMA, max_retries=1)
    assert result.data == {"name": "Bob", "age": 30}
    assert result.attempts == 2
    assert provider.chat.call_count == 2


def test_run_turn_combines_tools_and_schema_without_a_redundant_final_call():
    call = ToolCall(id="1", name="calculator", arguments={"a": 1, "b": 2, "operation": "add"})
    provider = _provider(
        ChatResponse(text="", tokens_out=5, tool_calls=[call]),
        ChatResponse(text='{"name": "Bob", "age": 30}', tokens_out=6, tool_calls=[]),
    )
    executor = ToolExecutor(get_tools(["calculator"]))

    result = run_turn(
        provider,
        _messages(),
        tools=[get_tools(["calculator"])[0].spec],
        tool_executor=executor,
        response_schema=SCHEMA,
    )

    assert result.data == {"name": "Bob", "age": 30}
    assert result.tool_call_count == 1
    # exactly 2 calls total: one tool-calling turn + one final turn re-used by
    # coerce_to_schema (no wasted extra call since it already validated)
    assert provider.chat.call_count == 2
    assert result.tokens_out == 11