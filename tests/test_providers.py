import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.core.errors import ErrorType, FormatError, GatewayError, ModelError
from src.providers.base import ChatMessage
from src.providers.groq_provider import GroqProvider
from src.providers.ollama_provider import OllamaProvider


def _msg():
    return [ChatMessage(role="user", content="hello")]


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


@patch("src.providers.ollama_provider.requests.post")
def test_ollama_connection_error_maps_to_model_error(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError("refused")

    provider = OllamaProvider(model="llama3.2")
    with pytest.raises(ModelError):
        provider.chat(_msg())


@patch("src.providers.ollama_provider.requests.post")
def test_ollama_bad_response_shape_maps_to_format_error(mock_post):
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"unexpected": "shape"}
    mock_post.return_value = mock_resp

    provider = OllamaProvider(model="llama3.2")
    with pytest.raises(FormatError):
        provider.chat(_msg())


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
