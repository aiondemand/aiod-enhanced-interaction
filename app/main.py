from beanie import init_beanie
import logfire
from motor.motor_asyncio import AsyncIOMotorClient
from functools import partial
from contextlib import asynccontextmanager
from threading import Thread

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models.asset_for_metadata_extraction import AssetForMetadataExtraction
from app.services.logging import setup_logger
from app.models.asset_collection import AssetCollection
from app.models.query import FilteredUserQuery, RecommenderUserQuery, SimpleUserQuery
from app.routers import filtered_sem_search as filtered_query_router
from app.routers import recommender_search as recommender_router
from app.routers import simple_sem_search as query_router
from app.routers import chatbot_endpoint as chatbot_router
from app.routers import healthcheck as healthcheck_router

from app.services.chatbot.website_scraper import scraping_wrapper
from app.services.threads.embedding_thread import compute_embeddings_for_aiod_assets_wrapper
from app.services.threads.milvus_gc_thread import delete_embeddings_of_aiod_assets_wrapper
from app.services.threads.threads import run_async_in_thread, start_async_thread
from app.services.threads.search_thread import QUERY_QUEUE, search_thread
from app.services.threads.db_gc_thread import mongo_cleanup
from app.services.threads.metadata_extraction_thread import extract_metadata_for_assets_wrapper

QUERY_THREAD: Thread | None = None
IMMEDIATE_EMB_THREAD: Thread | None = None
IMMEDIATE_CRAWLER_THREAD: Thread | None = None
IMMEDIATE_METADATA_EXTRACTION_THREAD: Thread | None = None
SCHEDULER: BackgroundScheduler | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    await app_init()
    yield
    await app_shutdown()


app = FastAPI(title="[AIoD] Enhanced Interaction", lifespan=lifespan)


app.include_router(healthcheck_router.router, prefix="", tags=["healthcheck"])
app.include_router(
    healthcheck_router.router, prefix=f"{settings.API_VERSION}", tags=["healthcheck"]
)

app.include_router(query_router.router, prefix="/query", tags=["query"])
app.include_router(query_router.router, prefix=f"{settings.API_VERSION}/query", tags=["query"])

if settings.CHATBOT.USE_CHATBOT:
    app.include_router(chatbot_router.router, prefix="/chatbot", tags=["chatbot"])
    app.include_router(
        chatbot_router.router, prefix=f"{settings.API_VERSION}/chatbot", tags=["chatbot"]
    )

app.include_router(recommender_router.router, prefix="/recommender", tags=["recommender_query"])
app.include_router(
    recommender_router.router,
    prefix=f"{settings.API_VERSION}/recommender",
    tags=["recommender_query"],
)

if settings.PERFORM_LLM_QUERY_PARSING:
    app.include_router(
        filtered_query_router.router, prefix=f"/filtered_query", tags=["filtered_query"]
    )
    app.include_router(
        filtered_query_router.router,
        prefix=f"{settings.API_VERSION}/filtered_query",
        tags=["filtered_query"],
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def setup_logfire() -> None:
    logfire.configure(
        token=settings.LOGFIRE_TOKEN, send_to_logfire="if-token-present", console=False
    )
    logfire.instrument_pydantic_ai()


async def app_init() -> None:
    setup_logger()
    setup_logfire()

    # Initialize MongoDB database
    app.db = await init_mongo_client()

    global QUERY_THREAD
    QUERY_THREAD = start_async_thread(search_thread)

    global SCHEDULER
    SCHEDULER = BackgroundScheduler()

    # Recurring AIoD updates
    SCHEDULER.add_job(
        partial(
            run_async_in_thread,
            target_func=partial(compute_embeddings_for_aiod_assets_wrapper, first_invocation=False),
        ),
        # Warning: We should not set the interval of recurring updates to a smaller
        # timespan than one day, otherwise some assets may be missed
        # Default is to perform the recurring update once per day
        CronTrigger(hour=0, minute=0),
    )
    # Recurring Milvus embedding cleanup
    SCHEDULER.add_job(
        partial(
            run_async_in_thread,
            target_func=delete_embeddings_of_aiod_assets_wrapper,
        ),
        CronTrigger(day=settings.AIOD.DAY_IN_MONTH_FOR_EMB_CLEANING, hour=0, minute=0),
    )
    # Recurring MongoDB cleanup
    SCHEDULER.add_job(
        partial(
            run_async_in_thread,
            target_func=mongo_cleanup,
        ),
        CronTrigger(hour=0, minute=0),
    )
    # Schedule to run daily at 2:00 AM to avoid conflicts with other jobs
    if settings.PERFORM_METADATA_EXTRACTION:
        SCHEDULER.add_job(
            partial(
                run_async_in_thread,
                target_func=extract_metadata_for_assets_wrapper,
            ),
            CronTrigger(hour=2, minute=0),
        )
    # Recurring scraping of AIoD websites
    if settings.CHATBOT.USE_CHATBOT:
        SCHEDULER.add_job(
            partial(
                run_async_in_thread,
                target_func=scraping_wrapper,
            ),
            CronTrigger(hour=0, minute=0),
        )
    SCHEDULER.start()

    # Immediate computation of AIoD asset embeddings
    global IMMEDIATE_EMB_THREAD
    IMMEDIATE_EMB_THREAD = start_async_thread(
        target_func=partial(compute_embeddings_for_aiod_assets_wrapper, first_invocation=True)
    )

    # Immediate crawling of AIoD websites
    global IMMEDIATE_CRAWLER_THREAD
    if settings.CHATBOT.USE_CHATBOT:
        IMMEDIATE_CRAWLER_THREAD = start_async_thread(target_func=scraping_wrapper)

    global IMMEDIATE_METADATA_EXTRACTION_THREAD
    if settings.PERFORM_METADATA_EXTRACTION:
        IMMEDIATE_METADATA_EXTRACTION_THREAD = start_async_thread(
            target_func=extract_metadata_for_assets_wrapper
        )


async def init_mongo_client() -> AsyncIOMotorClient:
    db = AsyncIOMotorClient(settings.MONGO.CONNECTION_STRING, uuidRepresentation="standard")[
        settings.MONGO.DBNAME
    ]
    # TODO multiprocessing_mode doesn't make the Database connection thread-safe
    # We need to move all the threading logic to a separate tasks of the primary thread instead
    # TODO replace embedding_thread, search_thread and recurring jobs for tasks
    # Github Issue: https://github.com/aiondemand/aiod-enhanced-interaction/issues/103
    await init_beanie(
        database=db,
        document_models=[
            AssetCollection,
<<<<<<< HEAD
            AssetForMetadataExtraction,
            SimpleUserQuery,
            FilteredUserQuery,
            RecommenderUserQuery,
=======
            SimpleUserQuery,
            FilteredUserQuery,
            RecommenderUserQuery,
            AssetForMetadataExtraction,
>>>>>>> main
        ],
        multiprocessing_mode=True,  # temporary patch
    )

    return db


async def app_shutdown() -> None:
    if QUERY_QUEUE:
        QUERY_QUEUE.put((None, None))
    if QUERY_THREAD:
        QUERY_THREAD.join(timeout=5)
    if IMMEDIATE_EMB_THREAD:
        IMMEDIATE_EMB_THREAD.join(timeout=5)
    if IMMEDIATE_CRAWLER_THREAD:
        IMMEDIATE_CRAWLER_THREAD.join(timeout=5)
    if IMMEDIATE_METADATA_EXTRACTION_THREAD:
        IMMEDIATE_METADATA_EXTRACTION_THREAD.join(timeout=5)
    if SCHEDULER:
        SCHEDULER.shutdown(wait=False)

    if getattr(app, "db", None) is not None:
        app.db.client.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000, host="localhost")
