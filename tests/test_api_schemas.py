"""Tests for src/api/schemas.py — the HTTP request/response DTOs.

Pure model tests — no FastAPI app, no network, no providers. These pin the
HTTP request contract (one message form per request; streaming is plain
chat only) and the response shape contract (transcript messages serialize
in the session-record vocabulary).
"""

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    MessageIn,
    MessageOut,
    StructuredRequest,
    StructuredResponse,
    ToolCallOut,
    UsageOut,
)


# ---------------------------------------------------------------------------
# ChatRequest: minimal parsing + CLI-matching defaults
# ---------------------------------------------------------------------------


class TestChatRequestDefaults:
    def test_minimal_prompt_request_parses(self):
        req = ChatRequest(provider="ollama", prompt="Hello")
        assert req.provider == "ollama"
        assert req.prompt == "Hello"

    def test_defaults_match_cli_chat_defaults(self):
        # CLI chat: --temperature 0.7, --max-tokens 512, stream off,
        # max-tool-iterations 8 (orchestrator), max-retries 2 (extractor).
        req = ChatRequest(provider="groq", prompt="Hi")
        assert req.temperature == 0.7
        assert req.max_tokens == 512
        assert req.stream is False
        assert req.max_tool_iterations == 8
        assert req.max_retries == 2

    def test_unset_optionals_are_none(self):
        req = ChatRequest(provider="groq", prompt="Hi")
        assert req.model is None
        assert req.timeout is None
        assert req.messages is None
        assert req.system is None
        assert req.tools is None
        assert req.schema_ is None
        assert req.session_path is None


# ---------------------------------------------------------------------------
# ChatRequest: one-of messages/prompt (D5)
# ---------------------------------------------------------------------------


class TestOneOfMessagesOrPrompt:
    def test_prompt_alone_is_valid(self):
        req = ChatRequest(provider="groq", prompt="Hi")
        assert req.prompt == "Hi"
        assert req.messages is None

    def test_messages_alone_is_valid(self):
        req = ChatRequest(
            provider="groq",
            messages=[
                MessageIn(role="system", content="You are terse."),
                MessageIn(role="user", content="Hi"),
            ],
        )
        assert len(req.messages) == 2
        assert req.prompt is None

    def test_both_messages_and_prompt_rejected(self):
        with pytest.raises(ValidationError) as excinfo:
            ChatRequest(provider="groq", prompt="Hi", messages=[MessageIn(role="user", content="Hi")])
        (message,) = [e["msg"] for e in excinfo.value.errors() if e["loc"] == ()]
        assert "either `messages`" in message

    def test_neither_messages_nor_prompt_rejected(self):
        with pytest.raises(ValidationError) as excinfo:
            ChatRequest(provider="groq")
        (message,) = [e["msg"] for e in excinfo.value.errors() if e["loc"] == ()]
        assert "One of `messages`" in message

    def test_empty_messages_list_rejected(self):
        # An empty list would silently mean "no history": force clients to
        # the prompt form for single-turn requests instead.
        with pytest.raises(ValidationError):
            ChatRequest(provider="groq", messages=[])


# ---------------------------------------------------------------------------
# ChatRequest: stream + tools/schema rejected (D8)
# ---------------------------------------------------------------------------


class TestStreamIsPlainChatOnly:
    def test_stream_with_tools_rejected_naming_the_conflict(self):
        with pytest.raises(ValidationError) as excinfo:
            ChatRequest(provider="groq", prompt="Hi", stream=True, tools=["calculator"])
        message = str(excinfo.value)
        assert "`tools`" in message
        assert "streaming is plain chat only" in message

    def test_stream_with_schema_rejected_naming_the_conflict(self):
        with pytest.raises(ValidationError) as excinfo:
            ChatRequest(provider="groq", prompt="Hi", stream=True, schema={"type": "object"})
        message = str(excinfo.value)
        assert "`schema`" in message
        assert "streaming is plain chat only" in message

    def test_stream_with_both_rejected_naming_both(self):
        with pytest.raises(ValidationError) as excinfo:
            ChatRequest(
                provider="groq",
                prompt="Hi",
                stream=True,
                tools=["calculator"],
                schema={"type": "object"},
            )
        message = str(excinfo.value)
        assert "`tools`" in message and "`schema`" in message

    def test_stream_alone_is_valid(self):
        req = ChatRequest(provider="groq", prompt="Hi", stream=True)
        assert req.stream is True

    def test_tools_without_stream_is_valid(self):
        req = ChatRequest(provider="groq", prompt="Hi", tools=["calculator"])
        assert req.tools == ["calculator"]

    def test_schema_without_stream_is_valid(self):
        schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
        req = ChatRequest(provider="groq", prompt="Hi", schema=schema)
        assert req.schema_ == schema


# ---------------------------------------------------------------------------
# `schema` alias round-trip (D7)
# ---------------------------------------------------------------------------


class TestSchemaAlias:
    def test_alias_accepts_schema_key_in_raw_json(self):
        raw = {
            "provider": "groq",
            "prompt": "Summarize",
            "schema": {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]},
        }
        req = ChatRequest.model_validate(raw)
        assert req.schema_ == raw["schema"]

    def test_alias_serializes_back_as_schema(self):
        schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
        req = ChatRequest(provider="groq", prompt="Hi", schema=schema)
        dumped = req.model_dump(by_alias=True, exclude_none=True)
        assert dumped["schema"] == schema
        assert "schema_" not in dumped

    def test_structured_request_requires_schema_via_alias(self):
        raw = {
            "provider": "groq",
            "text": "Ada wrote the first compiler.",
            "schema": {"type": "object", "properties": {"person": {"type": "string"}}, "required": ["person"]},
        }
        req = StructuredRequest.model_validate(raw)
        assert req.schema_ == raw["schema"]
        assert req.temperature == 0.0  # CLI structured default: determinism
        assert req.max_tokens == 512

    def test_structured_request_without_schema_rejected(self):
        with pytest.raises(ValidationError):
            StructuredRequest.model_validate({"provider": "groq", "text": "some text"})


# ---------------------------------------------------------------------------
# MessageIn: full-fidelity session-record shape accepted
# ---------------------------------------------------------------------------


class TestMessageIn:
    def test_plain_user_message(self):
        msg = MessageIn(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_assistant_message_with_tool_calls_round_trips_a_session_record(self):
        # Shape of a session-JSONL assistant record that requested a tool:
        # content may be None when the model only emitted tool calls.
        record = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "name": "calculator", "arguments": {"expression": "2+2"}}],
        }
        msg = MessageIn.model_validate(record)
        assert msg.role == "assistant"
        assert msg.content is None
        assert msg.tool_calls[0].id == "call_1"
        assert msg.tool_calls[0].arguments == {"expression": "2+2"}
        # Every field the session record sets survives validation and
        # re-serialization; the DTO additionally carries None for the
        # record's unset optional fields. (The exact record projection —
        # content always kept, other Nones omitted — is the mapper's job,
        # cross-checked against core.session._message_to_record in Step 2.)
        assert msg.model_dump() == {**record, "tool_call_id": None, "name": None}

    def test_tool_result_message_with_tool_call_id_and_name(self):
        record = {"role": "tool", "tool_call_id": "call_1", "name": "calculator", "content": "4"}
        msg = MessageIn.model_validate(record)
        assert msg.tool_call_id == "call_1"
        assert msg.name == "calculator"
        assert msg.model_dump(exclude_none=True) == record

    def test_unknown_role_rejected(self):
        with pytest.raises(ValidationError):
            MessageIn(role="bot", content="hi")

    def test_tool_call_arguments_must_be_an_object(self):
        with pytest.raises(ValidationError):
            MessageIn.model_validate(
                {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "name": "t", "arguments": "2+2"}]}
            )


# ---------------------------------------------------------------------------
# Response models: usage preference rules + transcript shape
# ---------------------------------------------------------------------------


class TestUsageOut:
    def test_provider_billed_tokens_in_has_no_method_label(self):
        usage = UsageOut(tokens_in=33, tokens_out=12)
        assert usage.token_count_method is None

    def test_client_side_fallback_carries_the_method_label(self):
        usage = UsageOut(tokens_in=17, tokens_out=12, token_count_method="tiktoken")
        assert usage.token_count_method == "tiktoken"

    def test_tokens_in_may_be_omitted(self):
        usage = UsageOut(tokens_out=5)
        assert usage.tokens_in is None


class TestResponseModelShapes:
    def test_chat_response_serializes_an_example_orchestration_result_shape(self):
        body = {
            "text": "42",
            "data": None,
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "2*21"},
                {"role": "assistant", "content": "42"},
            ],
            "provider": "groq",
            "model": "llama-3.1-8b-instant",
            "tool_call_count": 0,
            "tool_iterations": 0,
            "attempts": 1,
            "usage": {"tokens_in": 21, "tokens_out": 2},
        }
        out = ChatResponse.model_validate(body)
        assert out.text == "42"
        assert out.data is None
        assert [m.role for m in out.messages] == ["system", "user", "assistant"]
        assert out.usage == UsageOut(tokens_in=21, tokens_out=2)
        # Full-fidelity dump: the transcript is what a client feeds back as `messages`.
        assert out.model_dump(exclude_none=True)["messages"] == body["messages"]

    def test_structured_response_shape(self):
        out = StructuredResponse(
            data={"person": "Ada Lovelace"},
            raw_text='{"person": "Ada Lovelace"}',
            attempts=1,
            provider="groq",
            model="llama-3.1-8b-instant",
            usage=UsageOut(tokens_in=10, tokens_out=8, token_count_method=None),
        )
        dumped = out.model_dump(exclude_none=True)
        assert dumped["usage"] == {"tokens_in": 10, "tokens_out": 8}
        assert dumped["data"] == {"person": "Ada Lovelace"}

    def test_tool_call_out_serializes_transcript_tool_calls(self):
        out = ToolCallOut(id="call_1", name="calculator", arguments={"expression": "2+2"})
        assert out.model_dump() == {"id": "call_1", "name": "calculator", "arguments": {"expression": "2+2"}}

    def test_message_out_round_trips_a_tool_result_record(self):
        record = {"role": "tool", "tool_call_id": "call_1", "name": "calculator", "content": "4"}
        out = MessageOut.model_validate(record)
        assert out.model_dump(exclude_none=True) == record


class TestMessageOutSessionRecordShape:
    """MessageOut ⇄ session-JSONL record vocabulary (core.session).

    MessageOut is the HTTP DTO; the *record* shape (unset fields omitted,
    `content` kept even when None) is what a session file stores and what
    mappers.py produces from ChatMessages — cross-checked
    there against `_message_to_record` itself. Here we pin the DTO side:
    the field vocabulary is identical, so a client can round-trip a
    transcript received from /v1/chat back in as `messages`.
    """

    def test_message_out_accepts_and_round_trips_an_assistant_record(self):
        record = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "name": "calculator", "arguments": {"expression": "2+2"}}],
        }
        out = MessageOut.model_validate(record)
        assert out.role == "assistant"
        assert out.content is None  # None is preserved, not coerced to ""
        assert out.tool_calls[0].id == "call_1"
        assert out.tool_calls[0].arguments == {"expression": "2+2"}
        # DTO carries the record's fields plus Nones for its unset optionals;
        # the record projection itself is pinned via the mappers in Step 2.
        assert out.model_dump() == {**record, "tool_call_id": None, "name": None}

    def test_message_out_unset_optionals_default_to_none(self):
        out = MessageOut(role="user", content="Hi")
        assert out.tool_calls is None
        assert out.tool_call_id is None
        assert out.name is None
        # DTO dump keeps the None fields; the record shape omits them
        # (mappers.py owns that distinction, cross-checked in Step 2).
        assert out.model_dump() == {
            "role": "user",
            "content": "Hi",
            "tool_calls": None,
            "tool_call_id": None,
            "name": None,
        }
