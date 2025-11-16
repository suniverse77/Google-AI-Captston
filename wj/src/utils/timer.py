"""실행 시간 측정 유틸."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator, Optional


@contextmanager
def timing(section: str, logger=None) -> Iterator[float]:
    """컨텍스트 내 실행 시간을 로깅."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if logger:
        logger.info("%s took %.3f seconds", section, elapsed)


class Stopwatch:
    """구간별 시간 측정."""

    def __init__(self) -> None:
        self._start: Optional[float] = None

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> float:
        if self._start is None:
            raise RuntimeError("Stopwatch가 시작되지 않았습니다.")
        elapsed = time.perf_counter() - self._start
        self._start = None
        return elapsed


__all__ = ["timing", "Stopwatch"]

