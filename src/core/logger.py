"""Structured JSONL logging for every gateway request.

Each call to log_request() appends exactly one JSON object (one line) to
the log file, matching the schema required by the project spec:

    timestamp, provider, latency_ms, tokens_in, tokens_out,
    temperature, status, error_type, tool_calls, tool_iterations,
    plus additive fields added since the project spec (documented inline
    below): error_subtype and token_count_method (audit #3 / #13).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_LOG_PATH = Path(os.environ.get("LLM_GATEWAY_LOG_PATH", "logs/requests.jsonl"))


def log_request(
    *,
    provider: str,
    latency_ms: float,
    tokens_in: int,
    tokens_out: int,
    temperature: float,
    status: str,
    error_type: Optional[str] = None,
    error_subtype: Optional[str] = None,
    token_count_method: Optional[str] = None,
    tool_calls: Optional[int] = None,
    tool_iterations: Optional[int] = None,
    log_path: Path = DEFAULT_LOG_PATH,
) -> dict:
    """Append one structured JSON record for a request and return it.

    `status` must be "success" or "error". `error_type` should be one of
    the ErrorType values (see core.errors) when status == "error", else None.
    `error_subtype` should be the concrete exception class name when status
    == "error" (GatewayError subclasses know who they are), else None.
    `token_count_method` should be "tiktoken" or "heuristic" — how `tokens_in`
    was counted when client-side (provider-billed counts make it None), so
    cost/context analysis can tell precise counts from approximations (audit
    #13). Optional for backward compatibility with existing callers/tests.
    """

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "latency_ms": latency_ms,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "temperature": temperature,
        "status": status,
        "error_type": error_type,
        "error_subtype": error_subtype,
        "token_count_method": token_count_method,
        "tool_calls": tool_calls,
        "tool_iterations": tool_iterations,
    }

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return record