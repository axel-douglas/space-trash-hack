"""Execution backends for parallel candidate generation and scoring."""

from __future__ import annotations

import asyncio
import functools
import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Iterable, Sequence


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, ""))
    except (TypeError, ValueError):
        return default


DEFAULT_PARALLEL_THRESHOLD = _env_int("REXAI_PARALLEL_THRESHOLD", 4)


def max_workers_for(task_count: int) -> int:
    if task_count <= 1:
        return 1
    cpu = os.cpu_count() or 1
    return max(1, min(cpu, task_count))


class ExecutionBackend:
    """Simple protocol for execution backends used across the generator."""

    def __init__(self, max_workers: int = 1) -> None:
        self.max_workers = max(1, int(max_workers))

    # The map/submit/shutdown trio mirrors the ``concurrent.futures.Executor``
    # API but is intentionally lightweight so we can plug alternative
    # implementations (Ray, asyncio, distributed executors) without rewriting
    # call sites.
    def map(self, func: Callable[[Any], Any], iterable: Iterable[Any]) -> list[Any]:
        raise NotImplementedError

    def submit(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        raise NotImplementedError

    def shutdown(self) -> None:  # pragma: no cover - overriden by subclasses
        return None

    # Allow ``with backend:`` usage for lifecycle management.
    def __enter__(self) -> "ExecutionBackend":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.shutdown()


class _ImmediateFuture(Future):
    """Future compatible wrapper for synchronous backends."""

    def __init__(self, value: Any) -> None:
        super().__init__()
        self.set_result(value)


class SynchronousBackend(ExecutionBackend):
    def __init__(self) -> None:
        super().__init__(max_workers=1)

    def map(self, func: Callable[[Any], Any], iterable: Iterable[Any]) -> list[Any]:
        return [func(item) for item in iterable]

    def submit(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        result = func(*args, **kwargs)
        return _ImmediateFuture(result)


class ThreadPoolBackend(ExecutionBackend):
    def __init__(self, max_workers: int | None = None) -> None:
        workers = max_workers_for(max_workers or 0) if max_workers else max_workers_for(0)
        super().__init__(max_workers=workers)
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def map(self, func: Callable[[Any], Any], iterable: Iterable[Any]) -> list[Any]:
        return list(self._executor.map(func, iterable))

    def submit(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        return self._executor.submit(func, *args, **kwargs)

    def shutdown(self) -> None:
        self._executor.shutdown()


class AsyncioBackend(ExecutionBackend):
    """Asyncio powered backend that still exposes a synchronous API."""

    def __init__(self, max_workers: int | None = None) -> None:
        workers = max_workers or max_workers_for(0)
        super().__init__(max_workers=workers)
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

    async def _run_map(
        self,
        func: Callable[[Any], Any],
        items: Sequence[Any],
    ) -> list[Any]:
        loop = asyncio.get_running_loop()
        tasks = [loop.run_in_executor(self._executor, functools.partial(func, item)) for item in items]
        results: list[Any] = []
        for task in tasks:
            results.append(await task)
        return results

    def map(
        self, func: Callable[[Any], Any], iterable: Iterable[Any]
    ) -> list[Any] | asyncio.Task[list[Any]]:
        items = list(iterable)
        if not items:
            return []

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._run_map(func, items))

        return loop.create_task(self._run_map(func, items))

    def submit(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        return self._executor.submit(func, *args, **kwargs)

    def shutdown(self) -> None:
        self._executor.shutdown()


def create_backend(
    task_count: int,
    *,
    preferred: str | None = None,
    threshold: int = DEFAULT_PARALLEL_THRESHOLD,
    max_workers: int | None = None,
) -> ExecutionBackend:
    """Factory that produces an execution backend based on configuration."""

    name = preferred or os.getenv("REXAI_EXECUTION_BACKEND", "auto")
    name = str(name).strip().lower()

    if name in {"sync", "sequential", "serial"}:
        return SynchronousBackend()

    if name == "asyncio":
        workers = max_workers or max_workers_for(task_count)
        if task_count < threshold:
            workers = 1
        return AsyncioBackend(max_workers=workers)

    # Default behaviour covers "auto", "thread", "threads" and unknown labels.
    if task_count < threshold or name == "sync":
        return SynchronousBackend()

    workers = max_workers or max_workers_for(task_count)
    return ThreadPoolBackend(max_workers=workers)


__all__ = [
    "AsyncioBackend",
    "ExecutionBackend",
    "SynchronousBackend",
    "ThreadPoolBackend",
    "create_backend",
    "DEFAULT_PARALLEL_THRESHOLD",
    "max_workers_for",
]
