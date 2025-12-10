from threading import Thread
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app_temp.config import settings
from app_temp.job.asset_crawl import crawl_assets_job
from app_temp.models.asset_for_metadata_extraction import AssetForMetadataExtraction
from app_temp.services.logging import setup_logger
from app_temp.models.asset_collection import AssetCollection
from app_temp.services.threads import start_async_thread

IMMEDIATE_ASSET_CRAWL: Thread | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    await app_init()
    yield
    await app_shutdown()


app = FastAPI(title="[AIoD] Asset Retrieval", lifespan=lifespan)

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

    global IMMEDIATE_ASSET_CRAWL
    IMMEDIATE_ASSET_CRAWL = start_async_thread(target_func=crawl_assets_job)


async def init_mongo_client() -> AsyncIOMotorClient:
    db = AsyncIOMotorClient(settings.MONGO.CONNECTION_STRING, uuidRepresentation="standard")[
        settings.MONGO.DBNAME
    ]
    await init_beanie(
        database=db,
        document_models=[
            AssetCollection,
            AssetForMetadataExtraction,
        ],
        multiprocessing_mode=True,  # temporary patch
    )

    return db


async def app_shutdown() -> None:
    if IMMEDIATE_ASSET_CRAWL:
        IMMEDIATE_ASSET_CRAWL.join(timeout=5)

    if getattr(app, "db", None) is not None:
        app.db.client.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000, host="localhost")
