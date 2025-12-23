"""Celery tasks for search query processing."""

from __future__ import annotations

import asyncio
import logging
from typing import Type
from uuid import UUID

from app.celery_app import celery_app
from app.celery_tasks.search.helpers import (
    _ensure_worker_initialized,
    get_worker_instances,
    process_query_async,
)
from app.models.query import (
    BaseUserQuery,
    FilteredUserQuery,
    RecommenderUserQuery,
    SimpleUserQuery,
)


@celery_app.task(bind=True)
def process_query_task(self, query_id: str, query_type_name: str) -> dict:
    """
    Process a search query task.

    Args:
        query_id: UUID string of the query
        query_type_name: Name of the query type class (e.g., 'SimpleUserQuery')

    Returns:
        dict with task result information
    """
    # Ensure worker is initialized
    _ensure_worker_initialized()

    # Get worker-level instances
    _worker_model, _worker_embedding_store = get_worker_instances()

    # Map query type name to class
    query_type_map: dict[str, Type[BaseUserQuery]] = {
        "SimpleUserQuery": SimpleUserQuery,
        "FilteredUserQuery": FilteredUserQuery,
        "RecommenderUserQuery": RecommenderUserQuery,
    }
    query_type = query_type_map.get(query_type_name)

    # Run async function in sync context
    try:
        result = asyncio.run(
            process_query_async(UUID(query_id), query_type, _worker_model, _worker_embedding_store)
        )
        return result
    except Exception as e:
        logging.error(f"Error processing query {query_id}: {e}")
        return {"status": "failed", "error": str(e)}
