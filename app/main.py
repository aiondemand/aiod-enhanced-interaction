from beanie import init_beanie
import logfire
from motor.motor_asyncio import AsyncIOMotorClient
from functools import partial
import logging
from contextlib import asynccontextmanager
from threading import Thread

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models.asset_collection import AssetCollection
from app.models.query import FilteredUserQuery, RecommenderUserQuery, SimpleUserQuery
from app.routers import filtered_sem_search as filtered_query_router
from app.routers import recommender_search as recommender_router
from app.routers import simple_sem_search as query_router
from app.schemas.enums import SupportedAssetType
from app.services.metadata_filtering.metadata_extraction_agent import metadata_extractor_agent
from app.services.threads.embedding_thread import compute_embeddings_for_aiod_assets_wrapper
from app.services.threads.milvus_gc_thread import delete_embeddings_of_aiod_assets_wrapper
from app.services.threads.threads import run_async_in_thread, start_async_thread
from app.services.threads.search_thread import QUERY_QUEUE, search_thread
from app.services.threads.db_gc_thread import mongo_cleanup

QUERY_THREAD: Thread | None = None
IMMEDIATE_EMB_THREAD: Thread | None = None
SCHEDULER: BackgroundScheduler | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    await app_init()
    yield
    await app_shutdown()


app = FastAPI(title="[AIoD] Enhanced Search", lifespan=lifespan)


@app.get("/health", tags=["healthcheck"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(query_router.old_router, prefix="/query", tags=["query"])
app.include_router(query_router.old_router, prefix="/v1/query", tags=["query"], deprecated=True)
app.include_router(query_router.router, prefix="/v2/query", tags=["query"])

app.include_router(recommender_router.old_router, prefix="/recommender", tags=["recommender_query"])
app.include_router(
    recommender_router.old_router,
    prefix="/v1/recommender",
    tags=["recommender_query"],
    deprecated=True,
)
app.include_router(recommender_router.router, prefix="/v2/recommender", tags=["recommender_query"])

if settings.PERFORM_LLM_QUERY_PARSING:
    # Common endpoints across versions
    app.include_router(
        filtered_query_router.router, prefix="/experimental/filtered_query", tags=["filtered_query"]
    )
    app.include_router(
        filtered_query_router.router,
        prefix="/v1/experimental/filtered_query",
        tags=["filtered_query"],
        deprecated=True,
    )
    app.include_router(
        filtered_query_router.router,
        prefix="/v2/experimental/filtered_query",
        tags=["filtered_query"],
    )

    # GET endpoint that is different across versions
    app.include_router(filtered_query_router.router_diff, tags=["filtered_query"])


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def setup_logger() -> None:
    format_string = "%(asctime)s [%(levelname)s] %(name)s - %(message)s (%(filename)s:%(lineno)d)"
    logging.basicConfig(
        level=logging.INFO,
        format=format_string,
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )


def setup_logfire() -> None:
    logfire.configure(
        token=settings.LOGFIRE_TOKEN, send_to_logfire="if-token-present", console=False
    )
    logfire.instrument_pydantic_ai()


async def app_init() -> None:
    setup_logger()
    setup_logfire()

    example = """
        This dataset (license: XD=5-0) contains sensor and operational event data collected from a mid-scale industrial manufacturing line producing standardized metal components. Each sample corresponds to a single production cycle and includes continuous numeric readings gathered from temperature, pressure, and vibration sensors positioned along the conveyor, press, and finishing stations. Complementary metadata fields describe machine configuration parameters, such as feed rate, applied mechanical force settings, and lubricant application level. The dataset also includes a binary quality label indicating whether the produced component passed or failed final inspection testing.

The data was gathered over a period of approximately seven months under normal operating conditions, with periodic adjustments to machine setups as part of routine calibration and maintenance. Natural variation is present due to equipment wear, environmental temperature fluctuations, operator behavior, and production scheduling. No artificial perturbations or synthetic values were introduced.

This dataset is suitable for use in predictive maintenance research, anomaly detection modeling, process optimization studies, and supervised classification tasks focused on predicting product quality outcomes. Because the data reflects live industrial conditions, it may require preprocessing steps such as normalization, missing value handling, and filtering of short-duration noise spikes. The dataset does not include personally identifiable information.
    """

    out = await metadata_extractor_agent.extract_metadata(example, SupportedAssetType.DATASETS)

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
    SCHEDULER.start()

    # Immediate computation of AIoD asset embeddings
    global IMMEDIATE_EMB_THREAD
    IMMEDIATE_EMB_THREAD = start_async_thread(
        target_func=partial(compute_embeddings_for_aiod_assets_wrapper, first_invocation=True)
    )


async def init_mongo_client() -> AsyncIOMotorClient:
    db = AsyncIOMotorClient(settings.MONGO.connection_string, uuidRepresentation="standard")[
        settings.MONGO.DBNAME
    ]
    # TODO multiprocessing_mode doesn't make the Database connection thread-safe
    # We need to move all the threading logic to a separate tasks of the primary thread instead
    # TODO replace embedding_thread, search_thread and recurring jobs for tasks
    # Github Issue: https://github.com/aiondemand/aiod-enhanced-interaction/issues/103
    await init_beanie(
        database=db,
        document_models=[AssetCollection, SimpleUserQuery, FilteredUserQuery, RecommenderUserQuery],
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
    if SCHEDULER:
        SCHEDULER.shutdown(wait=False)

    if getattr(app, "db", None) is not None:
        app.db.client.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000, host="localhost")
