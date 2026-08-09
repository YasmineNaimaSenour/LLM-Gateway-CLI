"""Latency measurement helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class Timer:
    """A small stopwatch. Use via start()/stop() or as a context manager (measure_latency)."""

    _start: float = field(default=0.0, repr=False)
    _end: float = field(default=0.0, repr=False)

    def start(self) -> "Timer":
        self._start = time.perf_counter()
        self._end = self._start
        return self

    def stop(self) -> "Timer":
        self._end = time.perf_counter()
        return self

    @property # use property to enable lazy evaluation. that is, we only compute the elapsed time when it's accessed, and treat it as a read-only attribute.
    def elapsed_ms(self) -> float:
        """Milliseconds elapsed. Safe to read before stop() (returns running total)."""
        end = self._end if self._end >= self._start else time.perf_counter()
        return round((end - self._start) * 1000, 2)


# this is merely a wrapper around Timer
@contextmanager
def measure_latency() -> Iterator[Timer]:
    """Context manager yielding a Timer whose elapsed_ms is finalized on exit."""
    timer = Timer().start()
    try:
        yield timer
    finally:
        timer.stop()
