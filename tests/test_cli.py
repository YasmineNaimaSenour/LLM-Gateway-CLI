import json
from unittest.mock import MagicMock, patch

from src.cli import main
from src.core.types import ToolCall
from src.providers.base import ChatResponse

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
}


@patch("src.cli.log_request")
@patch("src.cli.build_provider")
def test_cli_success_path_returns_zero(mock_build_provider, mock_log_request):
    mock_provider = mock_build_provider.return_value
    mock_provider.chat.return_value.text = "hello back"
    mock_provider.chat.return_value.tokens_out = 5

    exit_code = main(["--provider", "ollama", "--prompt", "hi"])
    assert exit_code == 0
    assert mock_log_request.call_args.kwargs["status"] == "success"


@patch("src.cli.log_request")
@patch("src.cli.build_provider")
def test_cli_never_crashes_on_unexpected_exception(mock_build_provider, mock_log_request):
    mock_build_provider.side_effect = RuntimeError("something totally unexpected")

    exit_code = main(["--provider", "groq", "--prompt", "hi"])
    assert exit_code == 1  # reported as failure, but no exception propagates
    assert mock_log_request.call_args.kwargs["status"] == "error"


@patch("src.cli.log_request")
@patch("src.cli.build_provider")
def test_cli_explicit_chat_subcommand_matches_backward_compatible_flat_form(
    mock_build_provider, mock_log_request
):
    mock_provider = mock_build_provider.return_value
    mock_provider.chat.return_value.text = "hello back"
    mock_provider.chat.return_value.tokens_out = 5

    exit_code = main(["chat", "--provider", "ollama", "--prompt", "hi"])
    assert exit_code == 0
    assert mock_log_request.call_args.kwargs["status"] == "success"


@patch("src.cli.log_request")
@patch("src.cli.build_provider")
def test_cli_flat_form_still_works_but_prints_a_deprecation_warning(
    mock_build_provider, mock_log_request, capsys
):
    # The implicit 'chat' form still functions, but every use
    # prints a stderr warning so scripts get a visible migration signal.
    mock_provider = mock_build_provider.return_value
    mock_provider.chat.return_value = ChatResponse(text="hello back", tokens_out=5)

    exit_code = main(["--provider", "ollama", "--prompt", "hi"])

    assert exit_code == 0
    stderr = capsys.readouterr().err
    assert "implicit 'chat' subcommand is deprecated" in stderr
    assert mock_log_request.call_args.kwargs["status"] == "success"


@patch("src.cli.log_request")
@patch("src.cli.build_provider")
def test_cli_explicit_chat_form_prints_no_deprecation_warning(
    mock_build_provider, mock_log_request, capsys
):
    mock_provider = mock_build_provider.return_value
    mock_provider.chat.return_value = ChatResponse(text="hello back", tokens_out=5)

    exit_code = main(["chat", "--provider", "ollama", "--prompt", "hi"])

    assert exit_code == 0
    assert "deprecated" not in capsys.readouterr().err


# -- provider-reported prompt tokens in the log -----------------------------


@patch("src.cli.log_request")
@patch("src.cli.build_provider")
def test_cli_logs_provider_reported_tokens_in_when_available(mock_build_provider, mock_log_request):
    mock_provider = mock_build_provider.return_value
    mock_provider.chat.return_value = ChatResponse(text="hello back", tokens_out=5, tokens_in=33)

    exit_code = main(["chat", "--provider", "ollama", "--prompt", "hi"])

    assert exit_code == 0
    assert mock_log_request.call_args.kwargs["tokens_in"] == 33  # provider-billed, not tiktoken's guess


@patch("src.cli.log_request")
@patch("src.cli.build_provider")
def test_cli_falls_back_to_client_side_tokens_in_when_provider_reports_none(
    mock_build_provider, mock_log_request
):
    mock_provider = mock_build_provider.return_value
    mock_provider.chat.return_value = ChatResponse(text="hello back", tokens_out=5, tokens_in=None)

    exit_code = main(["chat", "--provider", "ollama", "--prompt", "hi"])

    assert exit_code == 0
    logged = mock_log_request.call_args.kwargs["tokens_in"]
    assert isinstance(logged, int) and logged > 0  # the pre-count, not None


# -- how the logged tokens_in was counted -----------------------------------


@patch("src.cli.log_request")
@patch("src.cli.build_provider")
def test_cli_logs_client_side_token_count_method_when_provider_reports_none(
    mock_build_provider, mock_log_request
):
    mock_provider = mock_build_provider.return_value
    mock_provider.chat.return_value = ChatResponse(text="hello back", tokens_out=5, tokens_in=None)

    exit_code = main(["chat", "--provider", "ollama", "--prompt", "hi"])

    assert exit_code == 0
    method = mock_log_request.call_args.kwargs["token_count_method"]
    assert method in ("tiktoken", "heuristic")  # whichever counter is active


@patch("src.cli.log_request")
@patch("src.cli.build_provider")
def test_cli_omits_token_count_method_for_provider_billed_tokens(
    mock_build_provider, mock_log_request
):
    mock_provider = mock_build_provider.return_value
    mock_provider.chat.return_value = ChatResponse(text="hello back", tokens_out=5, tokens_in=33)

    exit_code = main(["chat", "--provider", "ollama", "--prompt", "hi"])

    assert exit_code == 0
    assert mock_log_request.call_args.kwargs["token_count_method"] is None
    assert mock_log_request.call_args.kwargs["tokens_in"] == 33


# -- structured subcommand -------------------------------------------------


def _write(tmp_path, name, content):
    path = tmp_path / name
    if name.endswith(".json"):
        path.write_text(json.dumps(content), encoding="utf-8")
    else:
        path.write_text(content, encoding="utf-8")
    return str(path)


@patch("src.cli.log_request")
@patch("src.cli.build_provider")
def test_cli_structured_success_writes_json_and_logs_success(mock_build_provider, mock_log_request, tmp_path, capsys):
    schema_path = _write(tmp_path, "schema.json", SCHEMA)
    input_path = _write(tmp_path, "input.txt", "Bob is 30 years old.")

    mock_provider = mock_build_provider.return_value
    mock_provider.name = "ollama"
    mock_provider.chat.return_value = MagicMock(text='{"name": "Bob", "age": 30}', tokens_out=8)

    exit_code = main(["structured", "--provider", "ollama", "--input", input_path, "--schema", schema_path])

    assert exit_code == 0
    assert mock_log_request.call_args.kwargs["status"] == "success"
    printed = json.loads(capsys.readouterr().out)
    assert printed == {"name": "Bob", "age": 30}


@patch("src.cli.log_request")
@patch("src.cli.build_provider")
def test_cli_structured_writes_to_output_file_when_given(mock_build_provider, mock_log_request, tmp_path):
    schema_path = _write(tmp_path, "schema.json", SCHEMA)
    input_path = _write(tmp_path, "input.txt", "Bob is 30 years old.")
    output_path = tmp_path / "out.json"

    mock_provider = mock_build_provider.return_value
    mock_provider.name = "ollama"
    mock_provider.chat.return_value = MagicMock(text='{"name": "Bob", "age": 30}', tokens_out=8)

    exit_code = main(
        [
            "structured",
            "--provider",
            "ollama",
            "--input",
            input_path,
            "--schema",
            schema_path,
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"name": "Bob", "age": 30}


@patch("src.cli.log_request")
def test_cli_structured_invalid_schema_file_reports_error_without_calling_provider(mock_log_request, tmp_path):
    schema_path = tmp_path / "schema.json"
    schema_path.write_text("{not valid json", encoding="utf-8")
    schema_path = str(schema_path)
    input_path = _write(tmp_path, "input.txt", "Bob is 30 years old.")

    with patch("src.cli.build_provider") as mock_build_provider:
        exit_code = main(["structured", "--provider", "ollama", "--input", input_path, "--schema", schema_path])
        mock_build_provider.assert_not_called()

    assert exit_code == 1
    assert mock_log_request.call_args.kwargs["status"] == "error"
    assert mock_log_request.call_args.kwargs["error_type"] == "format"


@patch("src.cli.log_request")
def test_cli_structured_unsupported_schema_reports_error(mock_log_request, tmp_path):
    unsupported_schema = {"type": "object", "properties": {"a": {"$ref": "#/$defs/Foo"}}}
    schema_path = _write(tmp_path, "schema.json", unsupported_schema)
    input_path = _write(tmp_path, "input.txt", "some text")

    with patch("src.cli.build_provider") as mock_build_provider:
        exit_code = main(["structured", "--provider", "ollama", "--input", input_path, "--schema", schema_path])
        mock_build_provider.assert_not_called()

    assert exit_code == 1
    assert mock_log_request.call_args.kwargs["error_type"] == "format"


@patch("src.cli.log_request")
def test_cli_structured_missing_input_file_reports_error(mock_log_request, tmp_path):
    schema_path = _write(tmp_path, "schema.json", SCHEMA)

    exit_code = main(
        ["structured", "--provider", "ollama", "--input", str(tmp_path / "missing.txt"), "--schema", schema_path]
    )

    assert exit_code == 1
    assert mock_log_request.call_args.kwargs["status"] == "error"


@patch("src.cli.log_request")
@patch("src.cli.build_provider")
def test_cli_chat_with_schema_prints_validated_json(mock_build_provider, mock_log_request, tmp_path):
    schema_path = _write(tmp_path, "schema.json", SCHEMA)
    mock_provider = mock_build_provider.return_value
    mock_provider.name = "ollama"
    mock_provider.chat.return_value = ChatResponse(text='{"name": "Bob", "age": 30}', tokens_out=8)

    exit_code = main(["chat", "--provider", "ollama", "--prompt", "hi", "--schema", schema_path])

    assert exit_code == 0
    assert mock_log_request.call_args.kwargs["status"] == "success"


@patch("src.cli.log_request")
@patch("src.cli.build_provider")
def test_cli_chat_with_tools_executes_calculator_and_returns_final_answer(mock_build_provider, mock_log_request, capsys):
    mock_provider = mock_build_provider.return_value
    mock_provider.name = "ollama"
    tool_call = ToolCall(id="1", name="calculator", arguments={"a": 40, "b": 2, "operation": "add"})
    mock_provider.chat.side_effect = [
        ChatResponse(text="", tokens_out=5, tool_calls=[tool_call]),
        ChatResponse(text="It's 42.", tokens_out=4, tool_calls=[]),
    ]

    exit_code = main(["chat", "--provider", "ollama", "--prompt", "what is 40+2?", "--tools", "calculator"])

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "It's 42."
    assert mock_log_request.call_args.kwargs["tool_calls"] == 1
    assert mock_log_request.call_args.kwargs["tool_iterations"] == 1


@patch("src.cli.log_request")
@patch("src.cli.build_provider")
def test_cli_chat_with_tools_shows_loop_progress_on_stderr(
    mock_build_provider, mock_log_request, capsys
):
    # Tool-bearing turns can't stream, but the user shouldn't stare
    # at silence either — each provider round-trip and tool execution is
    # announced on stderr while stdout stays reserved for the final answer.
    mock_provider = mock_build_provider.return_value
    mock_provider.name = "ollama"
    tool_call = ToolCall(id="1", name="calculator", arguments={"a": 40, "b": 2, "operation": "add"})
    mock_provider.chat.side_effect = [
        ChatResponse(text="", tokens_out=5, tool_calls=[tool_call]),
        ChatResponse(text="It's 42.", tokens_out=4, tool_calls=[]),
    ]

    exit_code = main(["chat", "--provider", "ollama", "--prompt", "what is 40+2?", "--tools", "calculator"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "It's 42."  # answer alone on stdout
    assert "thinking (round 1)" in captured.err
    assert "calling tool: calculator" in captured.err
    assert "writing answer" in captured.err


@patch("src.cli.log_request")
def test_cli_chat_unknown_tool_reports_error_without_calling_provider(mock_log_request):
    with patch("src.cli.build_provider") as mock_build_provider:
        exit_code = main(["chat", "--provider", "ollama", "--prompt", "hi", "--tools", "not_a_real_tool"])
        mock_build_provider.assert_not_called()

    assert exit_code == 1
    assert mock_log_request.call_args.kwargs["error_type"] == "format"


@patch("src.cli.log_request")
@patch("src.cli.build_provider")
def test_cli_tool_loop_error_logs_its_own_error_subtype(mock_build_provider, mock_log_request):
    # A tool loop is FORMAT_ERROR in the 5-category taxonomy, but the log must
    # still distinguish it from a malformed schema/payload — via error_subtype.
    mock_provider = mock_build_provider.return_value
    mock_provider.name = "ollama"
    tool_call = ToolCall(id="1", name="calculator", arguments={"a": 40, "b": 2, "operation": "add"})
    mock_provider.chat.side_effect = [
        ChatResponse(text="", tokens_out=5, tool_calls=[tool_call]),
        ChatResponse(text="", tokens_out=5, tool_calls=[tool_call]),
        ChatResponse(text="", tokens_out=5, tool_calls=[tool_call]),
    ]

    exit_code = main(
        ["chat", "--provider", "ollama", "--prompt", "loop forever", "--tools", "calculator", "--max-tool-iterations", "2"]
    )

    assert exit_code == 1
    assert mock_log_request.call_args.kwargs["status"] == "error"
    assert mock_log_request.call_args.kwargs["error_type"] == "format"
    assert mock_log_request.call_args.kwargs["error_subtype"] == "ToolLoopError"


@patch("src.cli.log_request")
@patch("src.cli.build_provider")
def test_cli_stderr_and_log_record_name_the_same_error_class(
    mock_build_provider, mock_log_request, capsys
):
    # Pinned convention: stderr carries `[category:ClassName]` and the JSONL
    # record carries the same class in error_subtype. Both surfaces must say
    # the same thing at the same specificity — that agreement IS the
    # standardized convention, and it holds for every GatewayError because
    # both are emitted from the single _log_and_report choke point. Verified
    # here for a tool-loop failure and (below) an extraction failure.
    mock_provider = mock_build_provider.return_value
    mock_provider.name = "ollama"
    tool_call = ToolCall(id="1", name="calculator", arguments={"a": 40, "b": 2, "operation": "add"})
    mock_provider.chat.side_effect = [
        ChatResponse(text="", tokens_out=5, tool_calls=[tool_call]),
        ChatResponse(text="", tokens_out=5, tool_calls=[tool_call]),
        ChatResponse(text="", tokens_out=5, tool_calls=[tool_call]),
    ]

    exit_code = main(
        ["chat", "--provider", "ollama", "--prompt", "loop forever", "--tools", "calculator", "--max-tool-iterations", "2"]
    )

    assert exit_code == 1
    kwargs = mock_log_request.call_args.kwargs
    stderr = capsys.readouterr().err
    # stderr also carries tool-loop progress lines; find the error line.
    error_line = next(line for line in stderr.splitlines() if line.startswith("["))
    assert error_line.startswith(f"[{kwargs['error_type']}:{kwargs['error_subtype']}]")
    assert kwargs["error_subtype"] == "ToolLoopError"
    assert "[format:ToolLoopError]" in stderr


@patch("src.cli.log_request")
@patch("src.cli.build_provider")
def test_cli_stderr_and_log_agree_for_extraction_failures_too(
    mock_build_provider, mock_log_request, capsys, tmp_path
):
    schema_path = _write(tmp_path, "schema.json", SCHEMA)
    input_path = _write(tmp_path, "input.txt", "Bob.")
    mock_provider = mock_build_provider.return_value
    mock_provider.name = "ollama"
    mock_provider.chat.return_value = MagicMock(text="still not json", tokens_out=3)

    exit_code = main(
        [
            "structured", "--provider", "ollama",
            "--input", str(input_path), "--schema", str(schema_path),
            "--max-retries", "0",
        ]
    )

    assert exit_code == 1
    kwargs = mock_log_request.call_args.kwargs
    stderr = capsys.readouterr().err
    error_line = next(line for line in stderr.splitlines() if line.startswith("["))
    assert error_line.startswith(f"[{kwargs['error_type']}:{kwargs['error_subtype']}]")
    assert kwargs["error_subtype"] == "ExtractionError"
    assert "[format:ExtractionError]" in stderr


@patch("src.cli.log_request")
@patch("src.cli.build_provider")
def test_cli_structured_gives_up_after_max_retries_and_reports_error(
    mock_build_provider, mock_log_request, tmp_path
):
    schema_path = _write(tmp_path, "schema.json", SCHEMA)
    input_path = _write(tmp_path, "input.txt", "Bob.")

    mock_provider = mock_build_provider.return_value
    mock_provider.name = "ollama"
    mock_provider.chat.return_value = MagicMock(text="not json at all", tokens_out=3)

    exit_code = main(
        [
            "structured",
            "--provider",
            "ollama",
            "--input",
            input_path,
            "--schema",
            schema_path,
            "--max-retries",
            "1",
        ]
    )

    assert exit_code == 1
    assert mock_provider.chat.call_count == 2  # 1 initial + 1 retry
    assert mock_log_request.call_args.kwargs["status"] == "error"
    assert mock_log_request.call_args.kwargs["error_subtype"] == "ExtractionError"