import json
from unittest.mock import MagicMock, patch

from src.cli import main

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
