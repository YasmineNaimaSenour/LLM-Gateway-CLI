"""HTTP transport with transient-failure retry, shared by every provider.

What is retried, and why:

* ConnectionError  -> the request never completed; retrying cannot duplicate
  server-side work. (This also covers ConnectTimeout, which subclasses both
  ConnectionError and Timeout — failing to *establish* a connection is
  transient in exactly the same way.)
* 500/502/503/504  -> provider-side or gateway-side faults; by convention
  the client may replay the identical request.

What is deliberately NOT retried, and why:

* 429              -> rate limiting is per-request quota, not a fault; a
  blind immediate retry usually makes it worse. It already maps to
  RATE_LIMIT in the error taxonomy, where the caller can decide on an
  appropriate (much longer) backoff.
* other 4xx        -> permanent per-request failures (bad key, bad model,
  malformed payload); retrying verbatim can never succeed.
* other 5xx        -> not part of the convention above; classify-and-report
  is the honest outcome.
* requests.Timeout (read timeouts) -> the request may still be running
  server-side; a retry could double-bill or double-execute a tool call.
  Read timeouts propagate to the provider's own error mapping.

Backoff is exponential (0.5s, 1.0s, ...) with a small retry budget — enough
to ride out a blip, bounded enough to keep interactive CLI latency sane.
Every retry prints a one-line notice to stderr so the user is never left
watching a silent pause.
"""

from __future__ import annotations

import sys
import time
from typing import Callable, Dict, Optional

import requests

RETRYABLE_STATUS_CODES = frozenset({500, 502, 503, 504})
MAX_RETRIES = 2  # initial attempt + 2 retries
BASE_BACKOFF_SECONDS = 0.5
BACKOFF_MULTIPLIER = 2.0


def post_with_retry(
    url: str,
    *,
    payload: Optional[Dict] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float,
    stream: bool = False,
    max_retries: int = MAX_RETRIES,
    sleep: Optional[Callable[[float], None]] = None,
) -> requests.Response:
    """POST with retry on transient failures (see module docstring).

    Returns the first non-retryable response as-is; if retries are exhausted
    on a retryable status, the last response is returned and the caller's
    existing status handling decides what it means (a 500 still classifies
    and logs — retry only smooths over transients, it never hides them).
    ConnectionError is re-raised once retries are exhausted so the provider's
    existing ConnectionError mapping keeps working unchanged.

    `sleep` is injectable for tests; production callers use time.sleep.
    """
    do_sleep = sleep if sleep is not None else time.sleep

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout, stream=stream)
        except requests.exceptions.ConnectionError:
            if attempt < max_retries:
                _notice(f"connection failed", attempt, max_retries, do_sleep, _delay(attempt))
                continue
            raise  # exhausted: let the provider's ConnectionError mapping handle it

        if resp.status_code not in RETRYABLE_STATUS_CODES:
            return resp
        if attempt < max_retries:
            delay = _delay(attempt)
            _notice(f"got HTTP {resp.status_code}", attempt, max_retries, do_sleep, delay)
            continue
        return resp  # exhausted on a retryable status: hand it to the caller

    raise AssertionError("unreachable")  # the loop always returns or raises


def _delay(attempt: int) -> float:
    """Backoff after attempt N (0-based): 0.5s, 1.0s, 2.0s, ..."""
    return BASE_BACKOFF_SECONDS * (BACKOFF_MULTIPLIER**attempt)


def _notice(what: str, attempt: int, max_retries: int, do_sleep: Callable[[float], None], delay: float) -> None:
    """Tell the user why the CLI just paused, then back off."""
    print(
        f"[http-retry] transient error ({what}); retrying in {delay:.1f}s "
        f"(attempt {attempt + 2}/{max_retries + 1})...",
        file=sys.stderr,
    )
    do_sleep(delay)
