import json

from src.core.logger import log_request


def test_log_request_writes_valid_jsonl(tmp_path):
    log_path = tmp_path / "requests.jsonl"
    record = log_request(
        provider="ollama",
        latency_ms=123.4,
        tokens_in=10,
        tokens_out=20,
        temperature=0.7,
        status="success",
        error_type=None,
        log_path=log_path,
    )

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    parsed = json.loads(lines[0])
    assert parsed == record
    assert parsed["provider"] == "ollama"
    assert parsed["status"] == "success"


def test_log_request_appends(tmp_path):
    log_path = tmp_path / "requests.jsonl"
    for _ in range(3):
        log_request(
            provider="groq",
            latency_ms=10.0,
            tokens_in=1,
            tokens_out=1,
            temperature=0.0,
            status="error",
            error_type="rate_limit",
            log_path=log_path,
        )
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
