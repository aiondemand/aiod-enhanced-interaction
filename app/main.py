from beanie import init_beanie
import logfire
from motor.motor_asyncio import AsyncIOMotorClient
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.celery_tasks.maintenance.tasks import compute_embeddings_task, extract_metadata_task, scraping_task
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

    # Immediate Background tasks to execute
    compute_embeddings_task.delay(first_invocation=True)
    if settings.CHATBOT.USE_CHATBOT:
        scraping_task.delay()
    if settings.PERFORM_METADATA_EXTRACTION:
        extract_metadata_task.delay()


async def init_mongo_client() -> AsyncIOMotorClient:
    db = AsyncIOMotorClient(settings.MONGO.CONNECTION_STRING, uuidRepresentation="standard")[
        settings.MONGO.DBNAME
    ]
    await init_beanie(
        database=db,
        document_models=[
            AssetCollection,
            AssetForMetadataExtraction,
            SimpleUserQuery,
            FilteredUserQuery,
            RecommenderUserQuery,
        ],
        multiprocessing_mode=True,
    )
    
    return db


async def app_shutdown() -> None:
    if getattr(app, "db", None) is not None:
        app.db.client.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000, host="localhost")
