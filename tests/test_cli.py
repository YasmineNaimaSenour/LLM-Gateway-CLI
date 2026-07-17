from unittest.mock import patch

from src.cli import main


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
