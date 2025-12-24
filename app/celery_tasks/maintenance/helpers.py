"""Helper functions for maintenance tasks."""

from __future__ import annotations

import asyncio
import logging

from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

from app.config import settings
from app.models.asset_collection import AssetCollection
from app.models.asset_for_metadata_extraction import AssetForMetadataExtraction
from app.models.query import FilteredUserQuery, RecommenderUserQuery, SimpleUserQuery

# Track if MongoDB/Beanie has been initialized for this worker
_worker_db_initialized: bool = False


async def init_worker_db() -> None:
    """Initialize MongoDB/Beanie connection for worker process."""
    global _worker_db_initialized
    if _worker_db_initialized:
        return

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
    _worker_db_initialized = True
    logging.info("[CELERY WORKER] Initialized MongoDB/Beanie for maintenance worker")


def ensure_worker_db_initialized() -> None:
    """Ensure MongoDB/Beanie is initialized for this worker process."""
    global _worker_db_initialized
    if not _worker_db_initialized:
        asyncio.run(init_worker_db())
