import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.core.errors import ErrorType, FormatError, GatewayError, ModelError
from src.core.types import ToolSpec
from src.providers.base import ChatMessage
from src.providers.groq_provider import DEFAULT_GROQ_API_URL, GroqProvider
from src.providers.ollama_provider import DEFAULT_OLLAMA_TIMEOUT, OllamaProvider


def _msg():
    return [ChatMessage(role="user", content="hello")]


_TOOL = ToolSpec(name="calculator", description="add numbers", parameters={"type": "object", "properties": {}})


# -- ChatMessage content-only serialization (audit #22) -----------------------


def test_to_content_dict_carries_role_and_content_only():
    # The name is the contract (renamed from to_dict, audit #22): the
    # tool-calling fields — tool_calls, tool_call_id, name — are deliberately
    # NOT included. Token counting is the intended consumer; full-fidelity
    # serialization has its own explicit record format (src/core/session.py).
    message = ChatMessage(
        role="assistant",
        content="calling it",
        tool_calls=[],
        tool_call_id="call_1",
        name="calculator",
    )
    assert message.to_content_dict() == {"role": "assistant", "content": "calling it"}


def test_to_content_dict_maps_none_content_to_empty_string():
    # Assistant messages that carry only tool calls have None content; the
    # empty-string mapping matches what the wire format needs from a
    # content-only view.
    assert ChatMessage(role="assistant", content=None).to_content_dict() == {"role": "assistant", "content": ""}


# -- Ollama -------------------------------------------------------------------


@patch("src.providers.ollama_provider.requests.post")
def test_ollama_chat_success(mock_post):
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"message": {"content": "hi there"}, "prompt_eval_count": 12}
    mock_post.return_value = mock_resp

    provider = OllamaProvider(model="llama3.2")
    response = provider.chat(_msg())
    assert response.text == "hi there"
    assert response.tokens_out > 0
    assert response.tokens_in == 12  # provider-billed prompt count (audit #11)


@patch("src.providers.ollama_provider.requests.post")
def test_ollama_tokens_in_is_none_when_usage_not_reported(mock_post):
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"message": {"content": "hi there"}}
    mock_post.return_value = mock_resp

    provider = OllamaProvider(model="llama3.2")
    response = provider.chat(_msg())
    assert response.tokens_in is None  # callers fall back to client-side counting


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
        "usage": {"completion_tokens": 3, "prompt_tokens": 11},
    }
    mock_post.return_value = mock_resp

    provider = GroqProvider(model="llama-3.1-8b-instant")
    response = provider.chat(_msg())
    assert response.text == "hi"
    assert response.tokens_out == 3
    assert response.tokens_in == 11  # provider-billed prompt count (audit #11)


@patch("src.providers.groq_provider.requests.post")
def test_groq_tokens_in_is_none_when_usage_missing(mock_post, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"choices": [{"message": {"content": "hi"}}]}
    mock_post.return_value = mock_resp

    provider = GroqProvider(model="llama-3.1-8b-instant")
    response = provider.chat(_msg())
    assert response.tokens_in is None  # callers fall back to client-side counting


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


# -- interactive timeout + endpoint configurability (audit #23 / #24) ---------


def test_ollama_default_timeout_is_interactive_friendly():
    # Audit #23: 120s was a batch-appropriate default that left an interactive
    # user staring at a stuck server for two minutes. The default is now 60s;
    # OLLAMA_TIMEOUT is the documented override for batch/slow-hardware use.
    assert OllamaProvider().timeout == 60.0
    assert OllamaProvider().timeout == DEFAULT_OLLAMA_TIMEOUT


def test_ollama_timeout_precedence_argument_over_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_TIMEOUT", "300")
    assert OllamaProvider(timeout=15).timeout == 15.0  # explicit argument wins


def test_ollama_timeout_from_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_TIMEOUT", "300")
    assert OllamaProvider().timeout == 300.0


def test_ollama_timeout_accepts_string_env_values(monkeypatch):
    # Env vars are strings; the provider, not the caller, does the coercion.
    monkeypatch.setenv("OLLAMA_TIMEOUT", "90")
    assert OllamaProvider().timeout == 90.0


def test_groq_default_timeout_and_env_override(monkeypatch):
    monkeypatch.delenv("GROQ_TIMEOUT", raising=False)
    assert GroqProvider(model="m").timeout == 60.0
    monkeypatch.setenv("GROQ_TIMEOUT", "180")
    assert GroqProvider(model="m").timeout == 180.0
    assert GroqProvider(model="m", timeout=10).timeout == 10.0  # argument beats env


@patch("src.providers.groq_provider.requests.post")
def test_groq_posts_to_the_configured_endpoint(mock_post, monkeypatch):
    # Audit #24: api_url argument > GROQ_API_URL env > official endpoint.
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_API_URL", raising=False)
    mock_resp = MagicMock(status_code=200)
    mock_resp.json.return_value = {"choices": [{"message": {"content": "ok"}}], "usage": {}}
    mock_post.return_value = mock_resp

    GroqProvider(model="m").chat(_msg())
    assert mock_post.call_args.args[0] == "https://api.groq.com/openai/v1/chat/completions"

    mock_post.reset_mock()
    GroqProvider(model="m", api_url="http://127.0.0.1:9999/v1/").chat(_msg())
    assert mock_post.call_args.args[0] == "http://127.0.0.1:9999/v1"  # explicit, trailing / stripped

    monkeypatch.setenv("GROQ_API_URL", "http://proxy.example/v1")
    mock_post.reset_mock()
    GroqProvider(model="m").chat(_msg())
    assert mock_post.call_args.args[0] == "http://proxy.example/v1"  # env override