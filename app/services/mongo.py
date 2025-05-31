from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings
from app.models.asset_collection import AssetCollection
from app.models.query import FilteredUserQuery, RecommenderUserQuery, SimpleUserQuery


async def init_mongodb_client() -> AsyncIOMotorClient:
    db = AsyncIOMotorClient(str(settings.MONGODB.URI), uuidRepresentation="standard")[
        settings.MONGODB.DBNAME
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
