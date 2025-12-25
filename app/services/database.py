from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

from app.config import settings
from app.models.asset_collection import AssetCollection
from app.models.asset_for_metadata_extraction import AssetForMetadataExtraction
from app.models.query import SimpleUserQuery, FilteredUserQuery, RecommenderUserQuery


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
