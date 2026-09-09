import json
from datetime import datetime
from pathlib import Path

import pytest

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


def test_log_request_error_subtype_defaults_to_none_and_round_trips(tmp_path):
    log_path = tmp_path / "requests.jsonl"

    log_request(
        provider="ollama",
        latency_ms=1.0,
        tokens_in=1,
        tokens_out=1,
        temperature=0.0,
        status="success",
        error_type=None,
        log_path=log_path,
    )
    log_request(
        provider="ollama",
        latency_ms=1.0,
        tokens_in=1,
        tokens_out=1,
        temperature=0.0,
        status="error",
        error_type="format",
        error_subtype="ToolLoopError",
        log_path=log_path,
    )

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    success, failure = (json.loads(line) for line in lines)
    assert success["error_subtype"] is None
    assert failure["error_subtype"] == "ToolLoopError"


def test_log_request_includes_token_count_method_field(tmp_path):
    # audit #13: records state HOW tokens_in was counted, so analysis can
    # tell tiktoken from the heuristic fallback from provider-billed counts.
    log_path = tmp_path / "requests.jsonl"

    log_request(
        provider="ollama",
        latency_ms=1.0,
        tokens_in=7,
        tokens_out=3,
        temperature=0.0,
        status="success",
        token_count_method="heuristic",
        log_path=log_path,
    )
    log_request(
        provider="ollama",
        latency_ms=1.0,
        tokens_in=42,
        tokens_out=3,
        temperature=0.0,
        status="success",
        token_count_method=None,  # provider-billed count: method not applicable
        log_path=log_path,
    )

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    heuristic_record, provider_record = (json.loads(line) for line in lines)
    assert heuristic_record["token_count_method"] == "heuristic"
    assert provider_record["token_count_method"] is None


# -- logger hardening (audit #14) -------------------------------------------


def test_log_request_includes_tool_fields_when_set(tmp_path):
    log_path = tmp_path / "requests.jsonl"
    log_request(
        provider="ollama",
        latency_ms=10.0,
        tokens_in=5,
        tokens_out=9,
        temperature=0.7,
        status="success",
        tool_calls=2,
        tool_iterations=1,
        log_path=log_path,
    )
    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["tool_calls"] == 2
    assert record["tool_iterations"] == 1


def test_log_request_tool_fields_default_to_null_not_absent(tmp_path):
    # The fields are part of the record schema even when not applicable —
    # downstream analysis can rely on the key existing (as null).
    log_path = tmp_path / "requests.jsonl"
    log_request(
        provider="ollama",
        latency_ms=1.0,
        tokens_in=1,
        tokens_out=1,
        temperature=0.0,
        status="success",
        log_path=log_path,
    )
    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record["tool_calls"] is None
    assert record["tool_iterations"] is None


def test_log_request_timestamp_is_iso8601_and_timezone_aware(tmp_path):
    log_path = tmp_path / "requests.jsonl"
    log_request(
        provider="ollama",
        latency_ms=1.0,
        tokens_in=1,
        tokens_out=1,
        temperature=0.0,
        status="success",
        log_path=log_path,
    )
    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    parsed = datetime.fromisoformat(record["timestamp"])  # raises if not ISO
    assert parsed.tzinfo is not None  # UTC offset is present, so records sort unambiguously


def test_log_request_accepts_an_empty_provider_name(tmp_path):
    # The logger is policy-free: it serializes what it is given. (Callers —
    # the CLI — are responsible for provider validation via the registry.)
    log_path = tmp_path / "requests.jsonl"
    record = log_request(
        provider="",
        latency_ms=1.0,
        tokens_in=0,
        tokens_out=0,
        temperature=0.0,
        status="success",
        log_path=log_path,
    )
    round_tripped = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert round_tripped == record
    assert round_tripped["provider"] == ""


@pytest.fixture
def reload_logger_with_env(monkeypatch):
    """Reload src.core.logger under a controlled LLM_GATEWAY_LOG_PATH and
    restore the default (env-absent) state afterwards."""
    import importlib

    import src.core.logger as logger_module

    def _reload(value):
        if value is None:
            monkeypatch.delenv("LLM_GATEWAY_LOG_PATH", raising=False)
        else:
            monkeypatch.setenv("LLM_GATEWAY_LOG_PATH", value)
        importlib.reload(logger_module)
        return logger_module

    yield _reload
    monkeypatch.delenv("LLM_GATEWAY_LOG_PATH", raising=False)
    importlib.reload(logger_module)


def test_default_log_path_honors_the_env_var_override(reload_logger_with_env):
    # The module reads LLM_GATEWAY_LOG_PATH once at import: that override is
    # the supported way to relocate the log file without code changes.
    logger_module = reload_logger_with_env("/custom/place/logs.jsonl")
    assert logger_module.DEFAULT_LOG_PATH == Path("/custom/place/logs.jsonl")


def test_default_log_path_falls_back_to_logs_requests_jsonl(reload_logger_with_env):
    logger_module = reload_logger_with_env(None)
    assert logger_module.DEFAULT_LOG_PATH == Path("logs/requests.jsonl")


def _concurrent_writer(log_path_str: str, i: int) -> None:
    # Module-level so multiprocessing can pickle it; imports inside the worker.
    from src.core.logger import log_request

    log_request(
        provider=f"provider-{i % 3}",
        latency_ms=1.0,
        tokens_in=i,
        tokens_out=i,
        temperature=0.0,
        status="success",
        log_path=Path(log_path_str),
    )


def test_log_request_is_line_atomic_under_concurrent_writers(tmp_path):
    # Many processes appending simultaneously must never interleave partial
    # lines: every line stays parseable JSON and the total count is exact.
    import multiprocessing

    log_path = tmp_path / "concurrent.jsonl"
    workers, per_worker = 4, 5
    jobs = [(str(log_path), i) for i in range(workers * per_worker)]
    with multiprocessing.Pool(workers) as pool:
        pool.starmap(_concurrent_writer, jobs)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == workers * per_worker
    tokens_seen = set()
    for line in lines:
        record = json.loads(line)  # raises on torn/interleaved output
        tokens_seen.add(record["tokens_in"])
    assert tokens_seen == {i for i in range(workers * per_worker)}  # nothing lost or duplicated
