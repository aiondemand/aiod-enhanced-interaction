import logging
from contextlib import asynccontextmanager
from functools import partial
from threading import Thread
from time import sleep

from app.config import settings
from app.routers import simple_sem_search as query_router
from app.services.database import Database
from app.services.threads import threads
from app.services.threads.delete_thread import delete_embeddings_of_aiod_assets_wrapper
from app.services.threads.embedding_thread import (
    compute_embeddings_for_aiod_assets_wrapper,
)
from app.services.threads.search_thread import QUERY_QUEUE, search_thread
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

QUERY_THREAD: Thread | None = None
IMMEDIATE_EMB_THREAD: Thread | None = None
SCHEDULER: BackgroundScheduler | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_init()
    yield
    app_shutdown()


app = FastAPI(title="[AIoD] Semantic Search", lifespan=lifespan)

app.include_router(query_router.router, prefix="/query", tags=["query"])
app.include_router(
    query_router.router, prefix="/filtered_query", tags=["filtered_query"]
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def setup_logger():
    format_string = (
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s (%(filename)s:%(lineno)d)"
    )
    logging.basicConfig(
        level=logging.INFO,
        format=format_string,
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )


def app_init() -> None:
    sleep(10)  # Headstart for Milvus to fully initialize
    setup_logger()

    # Instantiate singletons before utilizing them in other threads
    Database()

    global QUERY_THREAD
    QUERY_THREAD = threads.start_async_thread(search_thread)

    global SCHEDULER
    SCHEDULER = BackgroundScheduler()

    # Recurring AIoD updates
    SCHEDULER.add_job(
        partial(
            threads.run_async_in_thread,
            target_func=partial(
                compute_embeddings_for_aiod_assets_wrapper, first_invocation=False
            ),
        ),
        # Warning: We should not set the interval of recurring updates to a smaller
        # timespan than one day, otherwise some assets may be missed
        # Default is to perform the recurring update once per day
        CronTrigger(hour=0, minute=0),
    )
    # Recurring Milvus embedding cleanup
    SCHEDULER.add_job(
        partial(
            threads.run_async_in_thread,
            target_func=delete_embeddings_of_aiod_assets_wrapper,
        ),
        CronTrigger(day=settings.AIOD.DAY_IN_MONTH_FOR_EMB_CLEANING, hour=0, minute=0),
    )
    SCHEDULER.start()

    # Immediate computation of AIoD asset embeddings
    global IMMEDIATE_EMB_THREAD
    IMMEDIATE_EMB_THREAD = threads.start_async_thread(
        target_func=partial(
            compute_embeddings_for_aiod_assets_wrapper, first_invocation=True
        )
    )


def app_shutdown() -> None:
    QUERY_QUEUE.put(None)
    QUERY_THREAD.join(timeout=5)
    IMMEDIATE_EMB_THREAD.join(timeout=5)
    SCHEDULER.shutdown(wait=False)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000, host="0.0.0.0")
