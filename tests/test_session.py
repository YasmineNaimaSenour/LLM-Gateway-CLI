import json

import pytest

from src.core.errors import SessionError
from src.core.session import (
    SessionStore,
    load_session_messages,
    save_session_messages,
)
from src.core.types import ToolCall
from src.providers.base import ChatMessage


def _roundtrip(store, messages):
    store.append(messages)
    return store.load()


def test_store_roundtrips_plain_messages(tmp_path):
    store = SessionStore(tmp_path / "session.jsonl")
    messages = [
        ChatMessage(role="system", content="You are terse."),
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="hello"),
    ]

    assert _roundtrip(store, messages) == messages


def test_store_roundtrips_full_tool_call_fidelity(tmp_path):
    store = SessionStore(tmp_path / "session.jsonl")
    messages = [
        ChatMessage(role="user", content="what is 40+2?"),
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id="call_1", name="calculator", arguments={"a": 40, "b": 2, "operation": "add"})],
        ),
        ChatMessage(role="tool", content="42.0", tool_call_id="call_1", name="calculator"),
        ChatMessage(role="assistant", content="It's 42."),
    ]

    assert _roundtrip(store, messages) == messages


def test_store_records_are_one_json_object_per_line(tmp_path):
    store = SessionStore(tmp_path / "session.jsonl")
    store.append([ChatMessage(role="user", content="hi"), ChatMessage(role="assistant", content="hello")])

    lines = (tmp_path / "session.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["role"] in ("user", "assistant") for line in lines)


def test_append_grows_file_without_truncating(tmp_path):
    store = SessionStore(tmp_path / "session.jsonl")
    first = [ChatMessage(role="user", content="turn 1")]
    second = [ChatMessage(role="assistant", content="reply 1"), ChatMessage(role="user", content="turn 2")]

    store.append(first)
    store.append(second)

    assert store.load() == first + second


def test_append_empty_list_is_a_noop_and_does_not_create_file(tmp_path):
    store = SessionStore(tmp_path / "nested" / "session.jsonl")

    store.append([])

    assert not store.exists()


def test_load_missing_file_returns_empty(tmp_path):
    assert SessionStore(tmp_path / "nope.jsonl").load() == []
    assert load_session_messages(str(tmp_path / "nope.jsonl")) is None


def test_load_skips_torn_trailing_line(tmp_path):
    path = tmp_path / "session.jsonl"
    good = json.dumps({"role": "user", "content": "hi"})
    torn = '{"role": "assistant", "content": "hel'  # process died mid-write
    path.write_text(f"{good}\n{torn}", encoding="utf-8")

    assert SessionStore(path).load() == [ChatMessage(role="user", content="hi")]


def test_load_skips_foreign_lines_but_keeps_valid_ones(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "hi"}),
                json.dumps({"role": "banana", "content": "not a real role"}),
                "not json at all",
            ]
        ),
        encoding="utf-8",
    )

    assert SessionStore(path).load() == [ChatMessage(role="user", content="hi")]


def test_load_file_with_no_valid_records_raises_session_error(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text("just some text\nnothing else\n", encoding="utf-8")

    with pytest.raises(SessionError):
        SessionStore(path).load()

    # it's a FormatError, so the CLI's taxonomy reports it as `format`
    from src.core.errors import FormatError

    assert issubclass(SessionError, FormatError)


def test_load_rejects_malformed_tool_call_arguments(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text(
        json.dumps({"role": "assistant", "content": "", "tool_calls": [{"id": "1", "name": "calc", "arguments": "40+2"}]}),
        encoding="utf-8",
    )

    with pytest.raises(SessionError):
        SessionStore(path).load()


def test_load_preserves_none_content_on_tool_call_messages(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text(
        json.dumps(
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "name": "calc", "arguments": {"a": 1}}]}
        ),
        encoding="utf-8",
    )

    loaded = SessionStore(path).load()
    assert loaded[0].content is None
    assert loaded[0].tool_calls[0].arguments == {"a": 1}


def test_save_session_messages_creates_file_on_first_turn(tmp_path):
    path = str(tmp_path / "sessions" / "demo.jsonl")  # parent dirs get created
    transcript = [
        ChatMessage(role="system", content="You are terse."),
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="hello"),
    ]

    save_session_messages(path, transcript)

    assert load_session_messages(path) == transcript


def test_save_session_messages_is_idempotent_for_same_transcript(tmp_path):
    path = str(tmp_path / "session.jsonl")
    transcript = [
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="hello"),
    ]

    save_session_messages(path, transcript)
    save_session_messages(path, transcript)  # e.g. a retried identical turn

    assert load_session_messages(path) == transcript


def test_save_session_messages_appends_only_new_tail_on_continuation(tmp_path):
    path = str(tmp_path / "session.jsonl")
    turn1 = [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="assistant", content="hello"),
    ]
    turn2 = turn1 + [ChatMessage(role="user", content="more"), ChatMessage(role="assistant", content="more reply")]

    save_session_messages(path, turn1)
    save_session_messages(path, turn2)

    assert load_session_messages(path) == turn2


def test_save_session_messages_rewrites_divergent_tail(tmp_path):
    """If on-disk history diverges from the transcript (e.g. the file was
    hand-edited), the common prefix is kept and everything after it is
    replaced by the transcript's version — append-only per line, but the
    semantic contract is 'file == transcript' after save."""
    path = str(tmp_path / "session.jsonl")
    base = [ChatMessage(role="user", content="hi")]
    saved = base + [ChatMessage(role="assistant", content="a different reply")]
    transcript = base + [ChatMessage(role="assistant", content="hello")]

    save_session_messages(path, saved)
    save_session_messages(path, transcript)

    loaded = load_session_messages(path)
    # prefix preserved, divergent messages appended (not deleted)
    assert loaded[:1] == base
    assert loaded[-1] == ChatMessage(role="assistant", content="hello")
    assert len(loaded) == 3  # user, stale assistant, corrected assistant


def test_session_store_roundtrip_via_temp_file_reuse_across_instances(tmp_path):
    """Simulates two CLI invocations: different SessionStore objects over the
    same file, load -> continue -> save."""
    path = str(tmp_path / "session.jsonl")

    turn1 = [ChatMessage(role="user", content="I'm Bob."), ChatMessage(role="assistant", content="Hi Bob.")]
    save_session_messages(path, turn1)

    prior = load_session_messages(path)
    turn2 = prior + [ChatMessage(role="user", content="what's my name?")]
    save_session_messages(path, turn2)

    assert load_session_messages(path) == turn2
