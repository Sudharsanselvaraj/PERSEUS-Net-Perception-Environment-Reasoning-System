"""
utils/timing.py
─────────────────────────────────────────────────────────────────
Lightweight timing utilities for pipeline performance monitoring.
"""

import time
import functools
from contextlib import contextmanager
from collections import deque
from typing import Deque, Dict
from utils.logger import get_logger

logger = get_logger(__name__)


class RollingTimer:
    """Track rolling average latency for a named operation."""

    def __init__(self, name: str, window: int = 30):
        self.name = name
        self._samples: Deque[float] = deque(maxlen=window)

    def record(self, elapsed_ms: float):
        self._samples.append(elapsed_ms)

    @property
    def avg_ms(self) -> float:
        return sum(self._samples) / len(self._samples) if self._samples else 0.0

    @property
    def fps(self) -> float:
        avg = self.avg_ms
        return 1000.0 / avg if avg > 0 else 0.0

    def __str__(self) -> str:
        return f"{self.name}: {self.avg_ms:.1f}ms avg ({self.fps:.1f} FPS)"


class PipelineProfiler:
    """Aggregate timer registry for the full pipeline."""

    def __init__(self):
        self._timers: Dict[str, RollingTimer] = {}

    def get_timer(self, name: str) -> RollingTimer:
        if name not in self._timers:
            self._timers[name] = RollingTimer(name)
        return self._timers[name]

    @contextmanager
    def measure(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - t0) * 1000
            self.get_timer(name).record(elapsed)

    def report(self) -> str:
        lines = ["── Pipeline Performance ──────────────────"]
        for name, timer in sorted(self._timers.items()):
            lines.append(f"  {timer}")
        return "\n".join(lines)


# Module-level singleton
profiler = PipelineProfiler()


def timed(name: str = None):
    """Decorator: records execution time of a function."""
    def decorator(func):
        label = name or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with profiler.measure(label):
                return func(*args, **kwargs)
        return wrapper
    return decorator
