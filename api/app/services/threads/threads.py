import asyncio
from threading import Thread
from typing import Callable, Coroutine


def run_async_in_thread(coroutine: Coroutine) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(coroutine)
    loop.close()


def start_async_thread(coroutine: Coroutine) -> Thread:
    thread = Thread(target=run_async_in_thread, args=(coroutine,))
    thread.start()
    return thread


def start_sync_thread(target_func: Callable[[], None]) -> Thread:
    thread = Thread(target=target_func)
    thread.start()
    return thread
