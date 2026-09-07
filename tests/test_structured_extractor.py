from unittest.mock import MagicMock

import pytest

from src.core.errors import ExtractionError
from src.providers.base import ChatMessage, ChatResponse
from src.structured.extractor import coerce_to_schema, extract, schema_instruction_message

SCHEMA = {
    "type": "object",
    "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"],
}


def _mock_provider(*response_texts, name="fake"):
    provider = MagicMock()
    provider.name = name
    provider.chat.side_effect = [MagicMock(text=t, tokens_out=len(t.split())) for t in response_texts]
    return provider


def test_extract_succeeds_on_clean_json_first_try():
    provider = _mock_provider('{"name": "Bob", "age": 30}')
    result = extract(provider, "Bob is 30.", SCHEMA)
    assert result.data == {"name": "Bob", "age": 30}
    assert result.attempts == 1
    assert provider.chat.call_count == 1


def test_extract_strips_markdown_code_fences():
    provider = _mock_provider('Sure, here you go:\n```json\n{"name": "Bob", "age": 30}\n```')
    result = extract(provider, "Bob is 30.", SCHEMA)
    assert result.data == {"name": "Bob", "age": 30}


def test_extract_finds_json_object_embedded_in_prose():
    provider = _mock_provider('The extracted record is {"name": "Bob", "age": 30} — hope that helps!')
    result = extract(provider, "Bob is 30.", SCHEMA)
    assert result.data == {"name": "Bob", "age": 30}


def test_extract_retries_after_unparsable_output_then_succeeds():
    provider = _mock_provider(
        'Sure! Here is the data: {"name": "Bob"',  # truncated / invalid JSON
        '{"name": "Bob", "age": 30}',
    )
    result = extract(provider, "Bob is 30.", SCHEMA, max_retries=2)
    assert result.data == {"name": "Bob", "age": 30}
    assert result.attempts == 2
    assert provider.chat.call_count == 2

    # the retry turn should include the failed attempt + a corrective follow-up
    second_call_messages = provider.chat.call_args_list[1][0][0]
    assert len(second_call_messages) == 4
    assert second_call_messages[-2].role == "assistant"
    assert second_call_messages[-1].role == "user"


def test_extract_retries_after_schema_violation_then_succeeds():
    provider = _mock_provider(
        '{"name": "Bob"}',  # valid JSON, but missing required 'age'
        '{"name": "Bob", "age": 30}',
    )
    result = extract(provider, "Bob is 30.", SCHEMA, max_retries=1)
    assert result.data == {"name": "Bob", "age": 30}
    assert result.attempts == 2


def test_extract_raises_extraction_error_after_exhausting_retries():
    provider = _mock_provider('{"name": "Bob"}', '{"name": "Bob"}', '{"name": "Bob"}')
    with pytest.raises(ExtractionError):
        extract(provider, "Bob.", SCHEMA, max_retries=2)
    assert provider.chat.call_count == 3  # 1 initial + 2 retries


def test_extract_raises_extraction_error_when_response_is_not_json_at_all():
    provider = _mock_provider("I cannot help with that request.")
    with pytest.raises(ExtractionError):
        extract(provider, "text", SCHEMA, max_retries=0)


def test_extract_rejects_non_object_json():
    provider = _mock_provider("[1, 2, 3]")
    with pytest.raises(ExtractionError):
        extract(provider, "text", SCHEMA, max_retries=0)


def test_extract_passes_temperature_and_max_tokens_through_to_provider():
    provider = _mock_provider('{"name": "Bob", "age": 30}')
    extract(provider, "text", SCHEMA, temperature=0.2, max_tokens=256)
    _, kwargs = provider.chat.call_args
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 256


def test_extract_forwards_schema_as_native_response_schema_hint():
    provider = _mock_provider('{"name": "Bob", "age": 30}')
    extract(provider, "text", SCHEMA)
    _, kwargs = provider.chat.call_args
    assert kwargs["response_schema"] == SCHEMA


def test_coerce_to_schema_reuses_initial_response_without_an_extra_call():
    provider = _mock_provider('{"name": "Bob", "age": 30}')
    messages = [ChatMessage(role="user", content="hi")]
    initial = ChatResponse(text='{"name": "Bob", "age": 30}', tokens_out=6)

    result = coerce_to_schema(provider, messages, SCHEMA, initial_response=initial)

    assert result.data == {"name": "Bob", "age": 30}
    assert result.attempts == 1
    provider.chat.assert_not_called()  # the reused response was enough


def test_coerce_to_schema_falls_back_to_a_fresh_call_when_initial_response_is_invalid():
    provider = _mock_provider('{"name": "Bob", "age": 30}')
    messages = [ChatMessage(role="user", content="hi")]
    initial = ChatResponse(text="not json at all", tokens_out=3)

    result = coerce_to_schema(provider, messages, SCHEMA, initial_response=initial, max_retries=1)

    assert result.data == {"name": "Bob", "age": 30}
    assert result.attempts == 2
    assert provider.chat.call_count == 1  # only the retry needed a fresh call


def test_schema_instruction_message_is_a_system_message_mentioning_the_schema():
    message = schema_instruction_message(SCHEMA)
    assert message.role == "system"
    assert '"age"' in message.content