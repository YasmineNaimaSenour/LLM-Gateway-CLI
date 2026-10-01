"""Tests for src/api/mappers.py — DTO ⇄ runtime vocabulary conversion.

Mapping tests: lossless dict ⇄ runtime conversion, the session-record
shape cross-check against core.session._message_to_record, the CLI's
token-preference rules, and the inline-schema pipeline. No FastAPI app,
no network — providers appear only as stubs returning canned responses.
"""

import pytest

from src.api.mappers import (
    build_usage_out,
    chat_message_to_message_out,
    extraction_result_to_structured_response,
    message_in_to_chat_message,
    messages_in_to_chat_messages,
    orchestration_result_to_chat_response,
    tool_call_in_to_tool_call,
    tool_call_to_tool_call_out,
    validate_inline_schema,
)
from src.api.schemas import (
    MessageIn,
    ToolCallIn,
    ToolCallOut,
    UsageOut,
)
from src.core.errors import SchemaError, UnsupportedSchemaError
from src.core.orchestrator import OrchestrationResult
from src.core.session import _message_from_record, _message_to_record
from src.core.types import ToolCall
from src.token_utils import count_method
from src.providers.base import ChatMessage, ChatResponse
from src.structured.extractor import ExtractionResult

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

ASSISTANT_WITH_TOOL_CALLS = ChatMessage(
    role="assistant",
    content=None,
    tool_calls=[ToolCall(id="call_1", name="calculator", arguments={"expression": "2+2"})],
)
TOOL_RESULT = ChatMessage(role="tool", tool_call_id="call_1", name="calculator", content="4")
PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": ["integer", "null"]},
    },
    "required": ["name"],
}


# ---------------------------------------------------------------------------
# Inbound mapping: MessageIn → ChatMessage
# ---------------------------------------------------------------------------


class TestInboundMappers:
    def test_plain_user_message_maps_losslessly(self):
        msg_in = MessageIn(role="user", content="Hello")
        msg = message_in_to_chat_message(msg_in)
        assert msg == ChatMessage(role="user", content="Hello")

    def test_system_message_with_content_maps_losslessly(self):
        msg = message_in_to_chat_message(MessageIn(role="system", content="You are terse."))
        assert msg.role == "system"
        assert msg.content == "You are terse."
        assert msg.tool_calls is None and msg.tool_call_id is None and msg.name is None

    def test_assistant_message_with_tool_calls_maps_losslessly(self):
        msg_in = MessageIn.model_validate(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "name": "calculator", "arguments": {"expression": "2+2"}}],
            }
        )
        msg = message_in_to_chat_message(msg_in)
        assert msg == ASSISTANT_WITH_TOOL_CALLS  # runtime dataclass equality: exact match
        assert msg.tool_calls[0] == ToolCall(id="call_1", name="calculator", arguments={"expression": "2+2"})

    def test_content_none_survives_the_round_trip(self):
        # Tool-call-only assistant messages carry content=None — the
        # mapper must not coerce it to "" (providers/token counting care).
        msg_in = MessageIn(role="assistant", content=None)
        msg = message_in_to_chat_message(msg_in)
        assert msg.content is None

    def test_tool_result_message_maps_id_and_name(self):
        msg_in = MessageIn.model_validate(
            {"role": "tool", "tool_call_id": "call_1", "name": "calculator", "content": "4"}
        )
        msg = message_in_to_chat_message(msg_in)
        assert msg == TOOL_RESULT

    def test_message_list_maps_in_order(self):
        msgs_in = [MessageIn(role="system", content="s"), MessageIn(role="user", content="u")]
        msgs = messages_in_to_chat_messages(msgs_in)
        assert msgs == [ChatMessage(role="system", content="s"), ChatMessage(role="user", content="u")]

    def test_tool_call_in_to_tool_call(self):
        tc = tool_call_in_to_tool_call(ToolCallIn(id="c9", name="current_time", arguments={"tz": "UTC"}))
        assert tc == ToolCall(id="c9", name="current_time", arguments={"tz": "UTC"})


# ---------------------------------------------------------------------------
# Outbound mapping: ChatMessage → MessageOut (+ session-record cross-check)
# ---------------------------------------------------------------------------


class TestOutboundMappers:
    def test_assistant_message_with_tool_calls_maps_losslessly(self):
        out = chat_message_to_message_out(ASSISTANT_WITH_TOOL_CALLS)
        assert out.role == "assistant"
        assert out.content is None
        assert out.tool_calls == [ToolCallOut(id="call_1", name="calculator", arguments={"expression": "2+2"})]
        assert out.tool_call_id is None and out.name is None

    def test_content_none_assistant_message_survives_the_round_trip(self):
        out = chat_message_to_message_out(ChatMessage(role="assistant", content=None))
        assert out.content is None

    def test_tool_result_message_maps_id_and_name(self):
        out = chat_message_to_message_out(TOOL_RESULT)
        assert out.role == "tool"
        assert out.tool_call_id == "call_1"
        assert out.name == "calculator"
        assert out.content == "4"

    def test_tool_call_to_tool_call_out(self):
        out = tool_call_to_tool_call_out(ToolCall(id="c1", name="calc", arguments={"x": 1}))
        assert out.model_dump() == {"id": "c1", "name": "calc", "arguments": {"x": 1}}

    # DTO vocabulary: the union of every field a session record can carry.
    DTO_VOCABULARY = {"role", "content", "tool_calls", "tool_call_id", "name"}

    def test_mappers_match_session_store_record_shape(self):
        """Shape guard: MessageOut must carry exactly the field
        vocabulary core.session._message_to_record stores, with identical
        values wherever the record speaks — pinning "messages out = what a
        session would store". The one deliberate difference: the DTO holds
        None for fields the record omits."""
        for message in [
            ChatMessage(role="system", content="You are terse."),
            ChatMessage(role="user", content="2+2"),
            ASSISTANT_WITH_TOOL_CALLS,
            ChatMessage(role="assistant", content="4"),
            TOOL_RESULT,
        ]:
            out = chat_message_to_message_out(message)
            record = _message_to_record(message)
            dumped = out.model_dump()
            # 1. No field outside the session-record vocabulary:
            assert set(dumped) == self.DTO_VOCABULARY
            # 2. Every field the record sets, the DTO carries identically
            #    (including content=None for tool-call-only assistants):
            for key, value in record.items():
                assert dumped[key] == value, f"field {key!r} diverges from the session record"
            # 3. Fields the record omits are exactly the DTO's Nones:
            for key in self.DTO_VOCABULARY - set(record):
                assert dumped[key] is None, f"{key!r} should be unset in the DTO"

    def test_message_out_to_chat_message_round_trip(self):
        """A transcript returned by the API can be reconstructed at the
        runtime boundary (this is the conversion Step 4/5 will use for
        `messages` in / the session loaders)."""
        out = chat_message_to_message_out(ASSISTANT_WITH_TOOL_CALLS)
        record = out.model_dump(exclude_none=True)
        # The session loader is the runtime's own record→ChatMessage parser.
        rebuilt = _message_from_record(record)
        assert rebuilt == ASSISTANT_WITH_TOOL_CALLS


# ---------------------------------------------------------------------------
# Usage mapping: CLI preference rules (D12)
# ---------------------------------------------------------------------------


class TestUsageMapper:
    def test_provider_billed_tokens_in_wins_with_no_method_label(self):
        usage = build_usage_out(provider_tokens_in=33, pre_count_tokens_in=30, tokens_out=12)
        assert usage == UsageOut(tokens_in=33, tokens_out=12, token_count_method=None)

    def test_none_provider_tokens_in_falls_back_to_pre_count_and_method(self):
        usage = build_usage_out(provider_tokens_in=None, pre_count_tokens_in=17, tokens_out=5)
        assert usage.tokens_in == 17
        assert usage.token_count_method in ("tiktoken", "heuristic")  # whatever count_method() reports
        assert usage.token_count_method == count_method()

    def test_zero_provider_count_still_wins_over_pre_count(self):
        # 0 is a real provider-reported value, not "missing": the
        # preference rule keys off None, not falsiness.
        usage = build_usage_out(provider_tokens_in=0, pre_count_tokens_in=30, tokens_out=1)
        assert usage.tokens_in == 0
        assert usage.token_count_method is None

    def test_no_counts_at_all(self):
        usage = build_usage_out(provider_tokens_in=None, pre_count_tokens_in=None, tokens_out=3)
        assert usage.tokens_in is None
        assert usage.token_count_method == count_method()


# ---------------------------------------------------------------------------
# Result → response mappers
# ---------------------------------------------------------------------------


class TestResultToResponseMappers:
    def _result(self, tokens_in=None):
        return OrchestrationResult(
            text="42",
            data=None,
            tokens_out=2,
            tool_call_count=0,
            tool_iterations=0,
            attempts=1,
            messages=[
                ChatMessage(role="system", content="Be terse."),
                ChatMessage(role="user", content="2*21"),
                ChatMessage(role="assistant", content="42"),
            ],
            tokens_in=tokens_in,
        )

    def test_orchestration_result_to_chat_response(self):
        resp = orchestration_result_to_chat_response(
            self._result(tokens_in=21), provider="groq", model="llama-3.1-8b-instant", pre_count_tokens_in=30
        )
        assert resp.text == "42"
        assert resp.data is None
        assert [m.role for m in resp.messages] == ["system", "user", "assistant"]
        assert resp.provider == "groq"
        assert resp.model == "llama-3.1-8b-instant"
        assert resp.tool_call_count == 0 and resp.tool_iterations == 0 and resp.attempts == 1
        assert resp.usage == UsageOut(tokens_in=21, tokens_out=2, token_count_method=None)

    def test_orchestration_result_falls_back_to_pre_count(self):
        resp = orchestration_result_to_chat_response(
            self._result(tokens_in=None), provider="ollama", model="llama3.2", pre_count_tokens_in=30
        )
        assert resp.usage.tokens_in == 30
        assert resp.usage.token_count_method is not None

    def test_transcript_mapping_does_not_mutate_the_result(self):
        result = self._result()
        before = [dataclasses.replace(m) for m in result.messages] if False else list(result.messages)
        orchestration_result_to_chat_response(result, provider="p", model="m", pre_count_tokens_in=1)
        assert result.messages == before  # serialized, never mutated

    def test_tool_bearing_transcript_carries_tool_fields(self):
        result = OrchestrationResult(
            text="4",
            data=None,
            tokens_out=1,
            tool_call_count=1,
            tool_iterations=1,
            attempts=1,
            messages=[ChatMessage(role="user", content="2+2"), ASSISTANT_WITH_TOOL_CALLS, TOOL_RESULT,
                      ChatMessage(role="assistant", content="4")],
        )
        resp = orchestration_result_to_chat_response(result, provider="p", model="m", pre_count_tokens_in=None)
        roles = [m.role for m in resp.messages]
        assert roles == ["user", "assistant", "tool", "assistant"]
        assert resp.messages[1].tool_calls[0].name == "calculator"
        assert resp.messages[2].tool_call_id == "call_1"
        assert resp.tool_call_count == 1 and resp.tool_iterations == 1

    def test_extraction_result_to_structured_response(self):
        result = ExtractionResult(
            data={"name": "Ada", "age": None},
            tokens_out=8,
            attempts=1,
            raw_text='{"name": "Ada", "age": null}',
            tokens_in=10,
        )
        resp = extraction_result_to_structured_response(
            result, provider="groq", model="llama-3.1-8b-instant", pre_count_tokens_in=14
        )
        assert resp.data == {"name": "Ada", "age": None}
        assert resp.raw_text == '{"name": "Ada", "age": null}'
        assert resp.attempts == 1
        assert resp.usage == UsageOut(tokens_in=10, tokens_out=8, token_count_method=None)

    def test_extraction_result_falls_back_to_pre_count(self):
        result = ExtractionResult(data={"name": "Ada"}, tokens_out=5, attempts=2, raw_text='{"name": "Ada"}')
        resp = extraction_result_to_structured_response(
            result, provider="ollama", model="llama3.2", pre_count_tokens_in=9
        )
        assert resp.usage.tokens_in == 9
        assert resp.usage.token_count_method is not None


# ---------------------------------------------------------------------------
# Inline schema validation (D7)
# ---------------------------------------------------------------------------


class TestValidateInlineSchema:
    def test_valid_schema_passes_through_unchanged(self):
        schema = dict(PERSON_SCHEMA)
        result = validate_inline_schema(schema)
        assert result == PERSON_SCHEMA  # same content...
        assert result is schema  # ...and the very same object: no rewriting

    def test_non_object_top_level_is_a_schema_error(self):
        with pytest.raises(SchemaError) as excinfo:
            validate_inline_schema(["not", "an", "object"])
        assert "JSON object at the top level" in str(excinfo.value)

    def test_invalid_json_schema_is_a_schema_error(self):
        with pytest.raises(SchemaError) as excinfo:
            validate_inline_schema({"type": "object", "properties": {"a": {"type": "strin"}}, "required": ["a"]})
        assert "Not a valid JSON Schema" in str(excinfo.value)

    def test_unsupported_feature_oneof_is_unsupported_with_path(self):
        schema = {
            "type": "object",
            "properties": {"x": {"oneOf": [{"type": "string"}, {"type": "integer"}]}},
            "required": ["x"],
        }
        with pytest.raises(UnsupportedSchemaError) as excinfo:
            validate_inline_schema(schema)
        assert "oneOf" in str(excinfo.value)
        assert "$.properties.x" in str(excinfo.value)  # path-qualified, like the CLI

    def test_root_not_object_is_unsupported(self):
        with pytest.raises(UnsupportedSchemaError) as excinfo:
            validate_inline_schema({"type": "string"})
        assert '"type": "object"' in str(excinfo.value)

    def test_nested_unsupported_feature_reports_nested_path(self):
        schema = {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "object", "properties": {"q": {"$ref": "#/x"}}}},
            },
            "required": ["items"],
        }
        with pytest.raises(UnsupportedSchemaError) as excinfo:
            validate_inline_schema(schema)
        assert "$.properties.items.items.properties.q" in str(excinfo.value)

    def test_schema_error_and_unsupported_are_distinct_failure_modes(self):
        # Both are FormatError subtypes, but callers (Step 3/4) must be able
        # to tell them apart — D11 maps them to different HTTP statuses.
        # A keyword-typed 'type' is broken JSON Schema (SchemaError); a valid
        # document using an unsupported keyword is a subset problem
        # (UnsupportedSchemaError).
        with pytest.raises(SchemaError):
            validate_inline_schema({"type": "object", "properties": {"a": {"type": 42}}, "required": ["a"]})
        with pytest.raises(UnsupportedSchemaError):
            validate_inline_schema({"type": "object", "properties": {"a": {"const": 1}}, "required": ["a"]})

