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

# Redis-based distributed locking
try:
    import redis
    from redis import Redis

    redis_client: Redis | None = None

    def get_redis_client() -> Redis:
        """Get or create Redis client for distributed locking."""
        global redis_client
        if redis_client is None:
            # Parse Redis URL from result backend
            result_backend = settings.CELERY.RESULT_BACKEND_URL
            redis_client = redis.from_url(result_backend)
        return redis_client

    def acquire_lock(lock_key: str, timeout: int = 3600) -> bool:
        """
        Acquire a distributed lock using Redis.
        Returns True if lock acquired, False otherwise.
        """
        client = get_redis_client()
        # SET key value NX EX timeout - set if not exists with expiration
        return bool(client.set(lock_key, "locked", nx=True, ex=timeout))

    def release_lock(lock_key: str) -> None:
        """Release a distributed lock."""
        client = get_redis_client()
        client.delete(lock_key)

except ImportError:
    # Fallback if redis is not available (shouldn't happen in production)
    logging.warning("Redis not available for distributed locking. Using simple check.")

    def acquire_lock(lock_key: str, timeout: int = 3600) -> bool:
        return True

    def release_lock(lock_key: str) -> None:
        pass


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
