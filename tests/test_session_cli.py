"""CLI-level tests for the multi-turn session feature (--session).

SessionStore unit tests live in test_session.py; this file covers the CLI
wiring: history loading, transcript persistence, failure semantics, and the
streaming / tools / schema / --system interactions.
"""

import json
from unittest.mock import patch

from src.cli import main
from src.core.session import load_session_messages
from src.core.types import ToolCall
from src.providers.base import ChatResponse

SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}


def _write_schema(tmp_path):
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(SCHEMA), encoding="utf-8")
    return str(path)


@patch("src.cli.log_request")
@patch("src.cli.build_provider")
def test_session_first_turn_creates_session_file(mock_build_provider, mock_log_request, tmp_path):
    session_path = str(tmp_path / "session.jsonl")
    mock_provider = mock_build_provider.return_value
    mock_provider.chat.return_value = ChatResponse(text="hello", tokens_out=2)

    exit_code = main(["chat", "--provider", "ollama", "--session", session_path, "--prompt", "hi"])

    assert exit_code == 0
    messages = load_session_messages(session_path)
    assert [m.role for m in messages] == ["user", "assistant"]
    assert [m.content for m in messages] == ["hi", "hello"]


@patch("src.cli.log_request")
@patch("src.cli.build_provider")
def test_session_second_turn_continues_conversation(mock_build_provider, mock_log_request, tmp_path):
    session_path = str(tmp_path / "session.jsonl")
    mock_provider = mock_build_provider.return_value
    mock_provider.chat.side_effect = [
        ChatResponse(text="first reply", tokens_out=2),
        ChatResponse(text="second reply", tokens_out=2),
    ]

    assert main(["chat", "--provider", "ollama", "--session", session_path, "--prompt", "hi"]) == 0
    assert main(["chat", "--provider", "ollama", "--session", session_path, "--prompt", "and more"]) == 0

    # the second invocation saw the first turn's transcript, not just its own prompt
    second_call_messages = mock_provider.chat.call_args_list[1][0][0]
    assert [m.content for m in second_call_messages] == ["hi", "first reply", "and more"]

    assert [m.content for m in load_session_messages(session_path)] == [
        "hi",
        "first reply",
        "and more",
        "second reply",
    ]


@patch("src.cli.log_request")
@patch("src.cli.build_provider")
def test_session_failed_turn_is_not_persisted(mock_build_provider, mock_log_request, tmp_path):
    session_path = str(tmp_path / "session.jsonl")
    mock_provider = mock_build_provider.return_value
    mock_provider.chat.side_effect = [
        ChatResponse(text="ok", tokens_out=1),
        RuntimeError("provider exploded"),
        ChatResponse(text="recovered", tokens_out=1),
    ]

    assert main(["chat", "--provider", "ollama", "--session", session_path, "--prompt", "one"]) == 0
    assert main(["chat", "--provider", "ollama", "--session", session_path, "--prompt", "two"]) == 1
    # retrying the same command resumes cleanly from turn 1's state
    assert main(["chat", "--provider", "ollama", "--session", session_path, "--prompt", "two"]) == 0

    contents = [m.content for m in load_session_messages(session_path)]
    assert contents == ["one", "ok", "two", "recovered"]


@patch("src.cli.log_request")
@patch("src.cli.build_provider")
def test_session_persists_tool_calls_and_results(mock_build_provider, mock_log_request, tmp_path):
    session_path = str(tmp_path / "session.jsonl")
    mock_provider = mock_build_provider.return_value
    mock_provider.name = "ollama"
    tool_call = ToolCall(id="1", name="calculator", arguments={"a": 40, "b": 2, "operation": "add"})
    mock_provider.chat.side_effect = [
        ChatResponse(text="", tokens_out=5, tool_calls=[tool_call]),
        ChatResponse(text="It's 42.", tokens_out=4, tool_calls=[]),
        # second turn: model answers directly, but the prior tool history is in context
        ChatResponse(text="Earlier you used the calculator.", tokens_out=4, tool_calls=[]),
    ]

    assert (
        main(
            [
                "chat",
                "--provider",
                "ollama",
                "--session",
                session_path,
                "--prompt",
                "what is 40+2?",
                "--tools",
                "calculator",
            ]
        )
        == 0
    )
    assert main(["chat", "--provider", "ollama", "--session", session_path, "--prompt", "what did you do?"]) == 0

    messages = load_session_messages(session_path)
    assert [m.role for m in messages] == ["user", "assistant", "tool", "assistant", "user", "assistant"]
    tool_msg = messages[2]
    assert tool_msg.content == "42.0"
    assert tool_msg.tool_call_id == "1"
    assert tool_msg.name == "calculator"
    # the continuation turn included the persisted tool exchange in its context
    third_call_messages = mock_provider.chat.call_args_list[2][0][0]
    assert [m.role for m in third_call_messages] == ["user", "assistant", "tool", "assistant", "user"]


@patch("src.cli.log_request")
@patch("src.cli.build_provider")
def test_session_keeps_system_message_and_ignores_it_on_continuation(
    mock_build_provider, mock_log_request, tmp_path, capsys
):
    session_path = str(tmp_path / "session.jsonl")
    mock_provider = mock_build_provider.return_value
    mock_provider.chat.side_effect = [
        ChatResponse(text="first", tokens_out=1),
        ChatResponse(text="second", tokens_out=1),
    ]

    assert (
        main(["chat", "--provider", "ollama", "--session", session_path, "--system", "You are terse.", "--prompt", "hi"])
        == 0
    )
    # --system on a continuation turn is a no-op; the session keeps its original
    assert (
        main(["chat", "--provider", "ollama", "--session", session_path, "--system", "You are loud.", "--prompt", "again"])
        == 0
    )

    second_call_messages = mock_provider.chat.call_args_list[1][0][0]
    assert second_call_messages[0].content == "You are terse."
    assert [m.role for m in second_call_messages] == ["system", "user", "assistant", "user"]
    assert "--system is ignored" in capsys.readouterr().err


@patch("src.cli.log_request")
@patch("src.cli.build_provider")
def test_session_schema_not_injected_twice_on_continuation(mock_build_provider, mock_log_request, tmp_path):
    schema_path = _write_schema(tmp_path)
    session_path = str(tmp_path / "session.jsonl")
    mock_provider = mock_build_provider.return_value
    mock_provider.name = "ollama"
    mock_provider.chat.side_effect = [
        ChatResponse(text='{"name": "Bob"}', tokens_out=4, tool_calls=[]),
        ChatResponse(text='{"name": "Bob"}', tokens_out=4, tool_calls=[]),
    ]

    assert main(["chat", "--provider", "ollama", "--session", session_path, "--prompt", "hi", "--schema", schema_path]) == 0
    assert (
        main(["chat", "--provider", "ollama", "--session", session_path, "--prompt", "again", "--schema", schema_path])
        == 0
    )

    second_call_messages = mock_provider.chat.call_args_list[1][0][0]
    system_contents = [m.content for m in second_call_messages if m.role == "system"]
    assert len(system_contents) == 1  # schema instruction present exactly once


@patch("src.cli.log_request")
@patch("src.cli.build_provider")
def test_session_streaming_turn_is_persisted(mock_build_provider, mock_log_request, tmp_path):
    session_path = str(tmp_path / "session.jsonl")
    mock_provider = mock_build_provider.return_value
    mock_provider.chat_stream.return_value = iter(["hel", "lo"])

    exit_code = main(["chat", "--provider", "ollama", "--session", session_path, "--prompt", "hi", "--stream"])

    assert exit_code == 0
    assert [m.content for m in load_session_messages(session_path)] == ["hi", "hello"]


@patch("src.cli.log_request")
def test_session_corrupt_file_reports_error_without_calling_provider(mock_log_request, tmp_path):
    session_path = tmp_path / "session.jsonl"
    session_path.write_text("this is not a session file\n", encoding="utf-8")

    with patch("src.cli.build_provider") as mock_build_provider:
        exit_code = main(["chat", "--provider", "ollama", "--session", str(session_path), "--prompt", "hi"])
        mock_build_provider.assert_not_called()

    assert exit_code == 1
    assert mock_log_request.call_args.kwargs["status"] == "error"
    assert mock_log_request.call_args.kwargs["error_type"] == "format"


@patch("src.cli.log_request")
@patch("src.cli.build_provider")
def test_session_save_failure_warns_but_does_not_fail_the_command(
    mock_build_provider, mock_log_request, tmp_path, capsys
):
    session_path = str(tmp_path / "session.jsonl")
    mock_provider = mock_build_provider.return_value
    mock_provider.chat.return_value = ChatResponse(text="hello", tokens_out=2)

    with patch("src.cli.save_session_messages", side_effect=OSError("disk full")):
        exit_code = main(["chat", "--provider", "ollama", "--session", session_path, "--prompt", "hi"])

    assert exit_code == 0  # the answer itself succeeded
    assert "could not update session file" in capsys.readouterr().err
