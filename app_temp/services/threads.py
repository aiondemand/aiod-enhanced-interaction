import asyncio
from threading import Thread
from typing import Any, Callable, Coroutine

# TODO Fix incorrect thread cleanup
# https://github.com/aiondemand/aiod-enhanced-interaction/issues/24


def run_async_in_thread(target_func: Callable[[], Coroutine[Any, Any, None]]) -> None:
    asyncio.run(target_func())


def start_async_thread(target_func: Callable[[], Coroutine[Any, Any, None]]) -> Thread:
    thread = Thread(target=run_async_in_thread, args=(target_func,))
    thread.start()
    return thread


def start_sync_thread(target_func: Callable[[], None]) -> Thread:
    thread = Thread(target=target_func)
    thread.start()
    return thread
