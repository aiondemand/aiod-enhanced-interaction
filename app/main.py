from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.celery_tasks import (
    compute_embeddings_task,
    extract_metadata_task,
    scraping_task,
)
from app import settings
from app.services.database import init_mongo_client
from app.services.logging import setup_logger
from app.routers import *


@asynccontextmanager
async def lifespan(app: FastAPI):
    await app_init()
    yield
    await app_shutdown()


app = FastAPI(title="[AIoD] Enhanced Interaction", lifespan=lifespan)


app.include_router(healthcheck_router, prefix="", tags=["healthcheck"])
app.include_router(healthcheck_router, prefix=f"{settings.API_VERSION}", tags=["healthcheck"])

app.include_router(query_router, prefix="/query", tags=["query"])
app.include_router(query_router, prefix=f"{settings.API_VERSION}/query", tags=["query"])

if settings.CHATBOT.USE_CHATBOT:
    app.include_router(chatbot_router, prefix="/chatbot", tags=["chatbot"])
    app.include_router(chatbot_router, prefix=f"{settings.API_VERSION}/chatbot", tags=["chatbot"])

app.include_router(recommender_router, prefix="/recommender", tags=["recommender_query"])
app.include_router(
    recommender_router,
    prefix=f"{settings.API_VERSION}/recommender",
    tags=["recommender_query"],
)

if settings.PERFORM_LLM_QUERY_PARSING:
    app.include_router(filtered_query_router, prefix=f"/filtered_query", tags=["filtered_query"])
    app.include_router(
        filtered_query_router,
        prefix=f"{settings.API_VERSION}/filtered_query",
        tags=["filtered_query"],
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def app_init() -> None:
    setup_logger()

    # Initialize MongoDB database
    app.db = await init_mongo_client()

    # Immediate Background tasks to execute
    compute_embeddings_task.delay(first_invocation=True)
    if settings.CHATBOT.USE_CHATBOT:
        scraping_task.delay()
    if settings.PERFORM_METADATA_EXTRACTION:
        extract_metadata_task.delay()


async def app_shutdown() -> None:
    if getattr(app, "db", None) is not None:
        app.db.client.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000, host="localhost")
