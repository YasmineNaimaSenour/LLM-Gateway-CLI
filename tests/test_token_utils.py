"""Tests for token counting utilities, including the count-method reporting
that feeds the `token_count_method` log field (audit #13)."""

from src.token_utils import (
    METHOD_HEURISTIC,
    METHOD_TIKTOKEN,
    _heuristic_token_count,
    count_method,
    count_message_tokens,
    count_tokens,
)


def test_count_method_reports_the_active_method():
    assert count_method() in (METHOD_TIKTOKEN, METHOD_HEURISTIC)


def test_method_labels_are_distinct():
    # The log field's two documented values must never collapse into one,
    # or "precise" and "approximate" counts would be indistinguishable.
    assert METHOD_TIKTOKEN != METHOD_HEURISTIC


def test_heuristic_fallback_counts_roughly_three_quarters_token_per_word():
    # Documents the fallback arithmetic: round(words / 0.75).
    assert _heuristic_token_count("one two three four") == 5


def test_counts_are_non_negative_and_empty_inputs_are_zero():
    assert count_tokens("") == 0
    assert count_tokens("hello") > 0
    assert count_message_tokens([]) == 0
    assert count_message_tokens([{"role": "user", "content": "hello"}]) > 0
