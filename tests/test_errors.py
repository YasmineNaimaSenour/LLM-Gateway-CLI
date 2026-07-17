from src.core.errors import (
    ErrorType,
    RateLimitError,
    classify_exception,
    to_gateway_error,
)


def test_classify_rate_limit_by_status_code():
    assert classify_exception(Exception("boom"), status_code=429) == ErrorType.RATE_LIMIT


def test_classify_rate_limit_by_message():
    assert classify_exception(Exception("Rate limit exceeded")) == ErrorType.RATE_LIMIT


def test_classify_context_overflow():
    assert classify_exception(Exception("maximum context length exceeded")) == ErrorType.CONTEXT_OVERFLOW


def test_classify_model_error_by_status():
    assert classify_exception(Exception("nope"), status_code=404) == ErrorType.MODEL_ERROR


def test_classify_format_error_by_status():
    assert classify_exception(Exception("nope"), status_code=400) == ErrorType.FORMAT_ERROR


def test_classify_unknown_default():
    assert classify_exception(Exception("something completely unexpected")) == ErrorType.UNKNOWN


def test_to_gateway_error_passthrough():
    original = RateLimitError("already normalized")
    assert to_gateway_error(original, provider="ollama") is original


def test_to_gateway_error_wraps_plain_exception():
    wrapped = to_gateway_error(Exception("429 too many requests"), provider="groq")
    assert wrapped.error_type == ErrorType.RATE_LIMIT
    assert wrapped.provider == "groq"
