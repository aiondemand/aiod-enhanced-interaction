"""Celery tasks for search query processing."""

import asyncio
from typing import Type
from uuid import UUID
import asyncio
from typing import Type, cast
from uuid import UUID
from celery.signals import worker_init

from app.config import settings
from app.models.query import (
    BaseUserQuery,
)
from app.services.database import init_mongo_client
from app.services.embedding_store import MilvusEmbeddingStore
from app.services.inference.model import AiModel
from app.celery_tasks.search.sem_search import (
    semantic_search_wrapper,
)
from app.celery_app import celery_app
from app.models.query import (
    BaseUserQuery,
    FilteredUserQuery,
    RecommenderUserQuery,
    SimpleUserQuery,
)

# Worker-level resources initialization, shared between all threads
_worker_model: AiModel | None = None
_worker_embedding_store: MilvusEmbeddingStore | None = None


# Hook executed when a worker (its main process) is initialized
@worker_init.connect
def ensure_worker_initialized(sender=None, conf=None, **kwargs) -> None:
    global _worker_model, _worker_embedding_store

    if str(sender).startswith(settings.CELERY.SEARCH_WORKER_NAME_PREFIX):
        asyncio.run(init_mongo_client())
        _worker_model = AiModel("cpu")
        _worker_embedding_store = MilvusEmbeddingStore()


@celery_app.task(bind=True, max_retries=3, acks_late=True, task_reject_on_worker_lost=True)
def search_query_task(self, query_id: str, query_type_name: str) -> dict:
    query_type_map: dict[str, Type[BaseUserQuery]] = {
        "SimpleUserQuery": SimpleUserQuery,
        "FilteredUserQuery": FilteredUserQuery,
        "RecommenderUserQuery": RecommenderUserQuery,
    }
    query_type = query_type_map.get(query_type_name, None)
    if query_type is None:
        raise ValueError(f"Invalid query type: {query_type_name}")

    model = cast(AiModel, _worker_model)
    embedding_store = cast(MilvusEmbeddingStore, _worker_embedding_store)

    return asyncio.run(semantic_search_wrapper(UUID(query_id), query_type, model, embedding_store))
