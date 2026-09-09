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
    # the returned transcript holds: user, assistant (tool_calls), tool result,
    # then the final assistant answer
    transcript = result.messages
    assert transcript[-3].role == "assistant"
    assert transcript[-2].role == "tool"
    assert transcript[-2].content == "3.0"
    assert transcript[-1].role == "assistant"
    assert transcript[-1].content == "The result is 3."


def test_run_turn_never_mutates_the_callers_messages_list():
    provider = _provider(ChatResponse(text="hello back", tokens_out=5))
    messages = _messages()

    result = run_turn(provider, messages)

    assert messages == [ChatMessage(role="user", content="hi")]
    assert result.messages is not messages
    assert result.messages == [ChatMessage(role="user", content="hi"), ChatMessage(role="assistant", content="hello back")]


def test_reusing_returned_messages_for_a_second_turn_does_not_double_history():
    provider = _provider(ChatResponse(text="first reply", tokens_out=2), ChatResponse(text="second reply", tokens_out=2))

    first = run_turn(provider, _messages())
    second = run_turn(provider, first.messages)

    assert provider.chat.call_count == 2
    # second call saw exactly the first turn's transcript — not a stale or doubled copy
    second_call_messages = provider.chat.call_args_list[1][0][0]
    assert [m.content for m in second_call_messages] == ["hi", "first reply"]
    assert [m.content for m in second.messages] == ["hi", "first reply", "second reply"]


def test_run_turn_appends_final_schema_valid_answer_to_returned_messages():
    provider = _provider(
        ChatResponse(text="not json at all", tokens_out=3),
        ChatResponse(text='{"name": "Bob", "age": 30}', tokens_out=6),
    )
    result = run_turn(provider, _messages(), response_schema=SCHEMA, max_retries=1)

    transcript = result.messages
    assert transcript[-1].role == "assistant"
    assert transcript[-1].content == '{"name": "Bob", "age": 30}'


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


# -- provider-reported prompt tokens (audit #11) ----------------------------


def test_run_turn_passes_provider_reported_tokens_in_through():
    provider = _provider(ChatResponse(text="hello back", tokens_out=5, tokens_in=37))
    result = run_turn(provider, _messages())
    assert result.tokens_in == 37


def test_run_turn_tokens_in_is_none_when_provider_does_not_report():
    provider = _provider(ChatResponse(text="hello back", tokens_out=5))
    result = run_turn(provider, _messages())
    assert result.tokens_in is None  # callers keep their client-side fallback


def test_run_turn_tool_loop_tokens_in_reflects_the_final_conversation_call():
    call = ToolCall(id="1", name="calculator", arguments={"a": 1, "b": 2, "operation": "add"})
    provider = _provider(
        ChatResponse(text="", tokens_out=5, tool_calls=[call], tokens_in=20),
        ChatResponse(text="The result is 3.", tokens_out=4, tokens_in=48),
    )
    executor = ToolExecutor(get_tools(["calculator"]))

    result = run_turn(provider, _messages(), tools=[get_tools(["calculator"])[0].spec], tool_executor=executor)

    # each successive call is billed on the full conversation so far, so the
    # provider's count from the LAST call is the accurate prompt total
    assert result.tokens_in == 48


def test_run_turn_schema_path_tokens_in_come_from_the_accepted_attempt():
    provider = _provider(
        ChatResponse(text="not json at all", tokens_out=3, tokens_in=15),
        ChatResponse(text='{"name": "Bob", "age": 30}', tokens_out=6, tokens_in=52),
    )
    result = run_turn(provider, _messages(), response_schema=SCHEMA, max_retries=1)
    assert result.tokens_in == 52  # the call that produced the accepted answer


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