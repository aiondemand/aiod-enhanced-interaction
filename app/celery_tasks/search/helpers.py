from __future__ import annotations

import asyncio
import logging
from typing import Type
from uuid import UUID

from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

from app.config import settings
from app.models.asset_collection import AssetCollection
from app.models.asset_for_metadata_extraction import AssetForMetadataExtraction
from app.models.query import (
    BaseUserQuery,
    FilteredUserQuery,
    RecommenderUserQuery,
    SimpleUserQuery,
)
from app.schemas.enums import QueryStatus
from app.services.embedding_store import MilvusEmbeddingStore
from app.services.inference.model import AiModel
from app.services.resilience import LocalServiceUnavailableException
from app.services.threads.search_thread import (
    fetch_user_query,
    search_across_assets_wrapper,
)

# Worker-level model and embedding store initialization
# These will be initialized once per worker process
_worker_model: AiModel | None = None
_worker_embedding_store: MilvusEmbeddingStore | None = None
_worker_db_initialized: bool = False


def _ensure_worker_initialized() -> None:
    """Ensure MongoDB/Beanie and model are initialized for this worker process."""
    global _worker_model, _worker_embedding_store, _worker_db_initialized

    # Initialize MongoDB/Beanie if not already done
    if not _worker_db_initialized:
        asyncio.run(_init_worker_db())
        _worker_db_initialized = True

    # Initialize model and embedding store if not already done (worker-level)
    if _worker_model is None:
        _worker_model = AiModel("cpu")
        logging.info("[CELERY WORKER] Initialized AiModel for search worker")

    if _worker_embedding_store is None:
        _worker_embedding_store = MilvusEmbeddingStore()
        logging.info("[CELERY WORKER] Initialized MilvusEmbeddingStore for search worker")


async def _init_worker_db() -> None:
    """Initialize MongoDB/Beanie connection for worker process."""
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
    logging.info("[CELERY WORKER] Initialized MongoDB/Beanie for search worker")


async def process_query_async(
    query_id: UUID,
    query_type: Type[BaseUserQuery],
    model: AiModel,
    embedding_store: MilvusEmbeddingStore,
) -> dict:
    """
    Async helper function to process a query.

    Args:
        query_id: UUID of the query to process
        query_type: Query type class
        model: AI model instance
        embedding_store: Embedding store instance

    Returns:
        dict with task result information
    """
    user_query = await fetch_user_query(query_id, query_type)
    if user_query is None:
        return {"status": "skipped", "reason": "Query not found or invalid"}

    logging.info(f"Searching relevant assets for query ID: {str(query_id)}")

    try:
        results = await search_across_assets_wrapper(model, embedding_store, user_query)
        user_query.result_set = results
        user_query.update_status(QueryStatus.COMPLETED)
        await user_query.replace_doc()

        return {
            "status": "completed",
            "query_id": str(query_id),
            "results_count": len(results.asset_ids) if results else 0,
        }
    except LocalServiceUnavailableException as e:
        logging.error(e)
        logging.error(
            "The above error has been encountered in the query processing task. "
            + "Task will be retried."
        )
        # Update query status to failed
        user_query.update_status(QueryStatus.FAILED)
        await user_query.replace_doc()
        # Re-raise to trigger Celery retry mechanism
        raise
    except Exception as e:
        user_query.update_status(QueryStatus.FAILED)
        await user_query.replace_doc()
        logging.error(e)
        logging.error(
            f"The above error has been encountered in the query processing task while processing query ID: {str(query_id)}"
        )
        return {"status": "failed", "error": str(e)}


def get_worker_instances() -> tuple[AiModel | None, MilvusEmbeddingStore | None]:
    """Get the worker-level model and embedding store instances."""
    return _worker_model, _worker_embedding_store
