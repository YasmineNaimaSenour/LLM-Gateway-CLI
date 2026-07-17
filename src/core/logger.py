"""Structured JSONL logging for every gateway request.

Each call to log_request() appends exactly one JSON object (one line) to
the log file, matching the schema required by the project spec:

    timestamp, provider, latency_ms, tokens_in, tokens_out,
    temperature, status, error_type
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
    log_path: Path = DEFAULT_LOG_PATH,
) -> dict:
    """Append one structured JSON record for a request and return it.

    `status` must be "success" or "error". `error_type` should be one of
    the ErrorType values (see core.errors) when status == "error", else None.
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
    }

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return record
