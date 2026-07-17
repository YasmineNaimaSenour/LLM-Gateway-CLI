from src.token_utils import count_message_tokens, count_tokens


def test_count_tokens_empty():
    assert count_tokens("") == 0


def test_count_tokens_positive_for_text():
    assert count_tokens("hello world, this is a test") > 0


def test_count_message_tokens_sums_all_messages():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi there!"},
    ]
    total = count_message_tokens(messages)
    assert total > count_tokens("You are helpful.") + count_tokens("Hi there!")
