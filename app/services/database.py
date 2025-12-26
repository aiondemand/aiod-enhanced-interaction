from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

from app import settings
import app.models as app_models


async def init_mongo_client() -> AsyncIOMotorClient:
    db = AsyncIOMotorClient(settings.MONGO.CONNECTION_STRING, uuidRepresentation="standard")[
        settings.MONGO.DBNAME
    ]

    db_models = [getattr(app_models, name) for name in app_models.__all__]

    await init_beanie(
        database=db,
        document_models=db_models,
        multiprocessing_mode=True,
    )

    return db
