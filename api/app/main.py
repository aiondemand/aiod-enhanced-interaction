import threading
from contextlib import asynccontextmanager
from functools import partial

from app.routers import query as query_router
from app.services.database import Database
from app.services.threads.embedding_thread import (
    compute_embeddings_for_AIoD_assets_wrapper,
)
from app.services.threads.search_thread import QUERY_QUEUE, start_search_thread
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

query_thread: threading.Thread | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_init()
    yield
    app_shutdown()


app = FastAPI(title="[AIoD] Semantic Search", lifespan=lifespan)

app.include_router(query_router.router, prefix="/query", tags=["query"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def app_init() -> None:
    # Instantiate singletons before utilizing them in other threads
    Database()

    global query_thread
    query_thread = start_search_thread()

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        compute_embeddings_for_AIoD_assets_wrapper, CronTrigger(hour=0, minute=0)
    )
    scheduler.start()

    threading.Thread(
        target=partial(
            compute_embeddings_for_AIoD_assets_wrapper, first_invocation=True
        )
    ).start()


def app_shutdown() -> None:
    QUERY_QUEUE.put(None)
    query_thread.join(timeout=5)

    # TODO stop the scheduler and interrupt all the jobs running in the background


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000, host="0.0.0.0")
