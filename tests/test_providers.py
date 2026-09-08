import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.core.errors import ErrorType, FormatError, GatewayError, ModelError
from src.core.types import ToolSpec
from src.providers.base import ChatMessage
from src.providers.groq_provider import GroqProvider
from src.providers.ollama_provider import OllamaProvider


def _msg():
    return [ChatMessage(role="user", content="hello")]


_TOOL = ToolSpec(name="calculator", description="add numbers", parameters={"type": "object", "properties": {}})


# -- Ollama -------------------------------------------------------------------


@patch("src.providers.ollama_provider.requests.post")
def test_ollama_chat_success(mock_post):
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"message": {"content": "hi there"}}
    mock_post.return_value = mock_resp

    provider = OllamaProvider(model="llama3.2")
    response = provider.chat(_msg())
    assert response.text == "hi there"
    assert response.tokens_out > 0


@patch("src.providers.http_utils.time.sleep")
@patch("src.providers.ollama_provider.requests.post")
def test_ollama_connection_error_maps_to_model_error(mock_post, mock_sleep):
    # post_with_retry retries the connection error before giving up; the
    # patched sleep keeps the backoff off the clock in tests.
    mock_post.side_effect = requests.exceptions.ConnectionError("refused")

    provider = OllamaProvider(model="llama3.2")
    with pytest.raises(ModelError):
        provider.chat(_msg())
    assert mock_post.call_count == 3  # initial attempt + 2 retries


@patch("src.providers.http_utils.time.sleep")
@patch("src.providers.ollama_provider.requests.post")
def test_ollama_retries_transient_503_then_succeeds(mock_post, mock_sleep):
    ok = MagicMock(status_code=200)
    ok.json.return_value = {"message": {"content": "recovered"}}
    mock_post.side_effect = [MagicMock(status_code=503), ok]

    provider = OllamaProvider(model="llama3.2")
    response = provider.chat(_msg())
    assert response.text == "recovered"
    assert mock_post.call_count == 2


@patch("src.providers.ollama_provider.requests.post")
def test_ollama_bad_response_shape_maps_to_format_error(mock_post):
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"unexpected": "shape"}
    mock_post.return_value = mock_resp

    provider = OllamaProvider(model="llama3.2")
    with pytest.raises(FormatError):
        provider.chat(_msg())


@patch("src.providers.ollama_provider.requests.post")
def test_ollama_parses_tool_calls_and_synthesizes_missing_id(mock_post):
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {
        "message": {
            "content": "",
            "tool_calls": [{"function": {"name": "calculator", "arguments": {"a": 1, "b": 2}}}],
        }
    }
    mock_post.return_value = mock_resp

    provider = OllamaProvider(model="llama3.2")
    response = provider.chat(_msg(), tools=[_TOOL])
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].name == "calculator"
    assert response.tool_calls[0].arguments == {"a": 1, "b": 2}
    assert response.tool_calls[0].id  # synthesized, non-empty

    sent_payload = mock_post.call_args.kwargs["json"]
    assert sent_payload["tools"][0]["function"]["name"] == "calculator"


@patch("src.providers.ollama_provider.requests.post")
def test_ollama_forwards_response_schema_into_format_field(mock_post):
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"message": {"content": "{}"}}
    mock_post.return_value = mock_resp

    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    provider = OllamaProvider(model="llama3.2")
    provider.chat(_msg(), response_schema=schema)

    sent_payload = mock_post.call_args.kwargs["json"]
    assert sent_payload["format"] == schema


# -- Groq -----------------------------------------------------------------------


def test_groq_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(GatewayError):
        GroqProvider(model="llama-3.1-8b-instant")


@patch("src.providers.groq_provider.requests.post")
def test_groq_chat_success(mock_post, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "hi"}}],
        "usage": {"completion_tokens": 3},
    }
    mock_post.return_value = mock_resp

    provider = GroqProvider(model="llama-3.1-8b-instant")
    response = provider.chat(_msg())
    assert response.text == "hi"
    assert response.tokens_out == 3


@patch("src.providers.groq_provider.requests.post")
def test_groq_rate_limit_maps_correctly(mock_post, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    mock_resp = MagicMock(status_code=429)
    mock_resp.json.return_value = {"error": {"message": "rate limit exceeded"}}
    mock_resp.text = json.dumps({"error": {"message": "rate limit exceeded"}})
    mock_post.return_value = mock_resp

    provider = GroqProvider(model="llama-3.1-8b-instant")
    with pytest.raises(GatewayError) as excinfo:
        provider.chat(_msg())
    assert excinfo.value.error_type == ErrorType.RATE_LIMIT


@patch("src.providers.groq_provider.requests.post")
def test_groq_parses_tool_calls_with_string_encoded_arguments(mock_post, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "function": {"name": "calculator", "arguments": '{"a": 1, "b": 2}'},
                        }
                    ],
                }
            }
        ],
        "usage": {"completion_tokens": 2},
    }
    mock_post.return_value = mock_resp

    provider = GroqProvider(model="llama-3.1-8b-instant")
    response = provider.chat(_msg(), tools=[_TOOL])
    assert response.tool_calls[0].id == "call_abc123"
    assert response.tool_calls[0].arguments == {"a": 1, "b": 2}

    sent_payload = mock_post.call_args.kwargs["json"]
    assert sent_payload["tools"][0]["function"]["name"] == "calculator"


@patch("src.providers.groq_provider.requests.post")
def test_groq_raises_format_error_on_malformed_tool_call_arguments(mock_post, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [{"id": "call_1", "function": {"name": "calculator", "arguments": "{not json"}}],
                }
            }
        ],
        "usage": {},
    }
    mock_post.return_value = mock_resp

    provider = GroqProvider(model="llama-3.1-8b-instant")
    with pytest.raises(FormatError):
        provider.chat(_msg(), tools=[_TOOL])


@patch("src.providers.groq_provider.requests.post")
def test_groq_forwards_response_schema_as_json_object_mode(mock_post, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"choices": [{"message": {"content": "{}"}}], "usage": {}}
    mock_post.return_value = mock_resp

    provider = GroqProvider(model="llama-3.1-8b-instant")
    provider.chat(_msg(), response_schema={"type": "object", "properties": {}})

    sent_payload = mock_post.call_args.kwargs["json"]
    assert sent_payload["response_format"] == {"type": "json_object"}