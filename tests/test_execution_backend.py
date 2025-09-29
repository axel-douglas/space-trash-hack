import asyncio
import time


from app.modules import execution


def test_thread_backend_outpaces_synchronous_execution():
    tasks = 4
    sleep_time = 0.05

    def work(_idx: int) -> float:
        time.sleep(sleep_time)
        return time.perf_counter()

    thread_backend = execution.ThreadPoolBackend(max_workers=tasks)
    try:
        start_parallel = time.perf_counter()
        thread_backend.map(work, range(tasks))
        elapsed_parallel = time.perf_counter() - start_parallel
    finally:
        thread_backend.shutdown()

    sync_backend = execution.SynchronousBackend()
    start_sync = time.perf_counter()
    sync_backend.map(work, range(tasks))
    elapsed_sync = time.perf_counter() - start_sync

    assert elapsed_parallel < elapsed_sync * 0.75
    assert elapsed_parallel < sleep_time * tasks


def test_asyncio_backend_map_from_running_loop():
    backend = execution.AsyncioBackend()

    async def _invoke_map() -> list[int]:
        result = backend.map(lambda x: x + 1, range(5))
        assert isinstance(result, list)
        return result

    try:
        result = asyncio.run(_invoke_map())
    finally:
        backend.shutdown()

    assert result == [1, 2, 3, 4, 5]
