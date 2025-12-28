"""Celery tasks for search query processing."""

import asyncio
from typing import Type
from uuid import UUID
import asyncio
from typing import Type, cast
from uuid import UUID
from celery.signals import worker_init

from mistralai import Mistral
from pymilvus import MilvusClient

from app import settings
from app.models import FilteredUserQuery, RecommenderUserQuery, SimpleUserQuery
from app.models.query import BaseUserQuery
from app.services.database import init_mongo_client
from app.services.embedding_store import EmbeddingStore, MilvusEmbeddingStore
from app.services.inference.model import AiModel
from app.celery_tasks.search.sem_search import (
    semantic_search_wrapper,
)
from app.celery_app import celery_app
from app.services.chatbot.chatbot import ChatbotService

# Worker-level resources initialization, shared between all threads
_worker_model: AiModel | None = None
_worker_embedding_store: EmbeddingStore | None = None
_worker_chatbot_service: ChatbotService | None = None


# Hook executed when a worker (its main process) is initialized
@worker_init.connect
def initialize_main_search_worker_process(sender=None, conf=None, **kwargs) -> None:
    global _worker_model, _worker_embedding_store, _worker_chatbot_service

    if str(sender).startswith(settings.CELERY.SEARCH_WORKER_NAME_PREFIX):
        asyncio.run(init_mongo_client())
        _worker_model = AiModel("cpu")
        _worker_embedding_store = MilvusEmbeddingStore()

        # Initialize chatbot service if chatbot is enabled
        if settings.CHATBOT.USE_CHATBOT:
            milvus_client = MilvusClient(
                uri=settings.MILVUS.HOST, token=settings.MILVUS.MILVUS_TOKEN
            )
            mistral_client = Mistral(api_key=settings.CHATBOT.MISTRAL_KEY)
            _worker_chatbot_service = ChatbotService(
                mistral_client=mistral_client,
                milvus_client=milvus_client,
                embedding_model=_worker_model,
            )


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
    embedding_store = cast(EmbeddingStore, _worker_embedding_store)

    return asyncio.run(semantic_search_wrapper(UUID(query_id), query_type, model, embedding_store))


@celery_app.task(bind=True, max_retries=3, acks_late=True, task_reject_on_worker_lost=True)
def chatbot_conversation_task(self, user_query: str, conversation_id: str | None = None) -> dict:
    """
    Process a chatbot conversation request.

    Returns:
        dict with 'content' and 'conversation_id'
    """
    chatbot_service = cast(ChatbotService, _worker_chatbot_service)
    return chatbot_service.process_conversation(user_query, conversation_id)


@celery_app.task(bind=True, max_retries=3, acks_late=True, task_reject_on_worker_lost=True)
def chatbot_history_task(self, conversation_id: str) -> dict:
    """
    Retrieve conversation history.

    Returns:
        dict representing the conversation history
    """
    chatbot_service = cast(ChatbotService, _worker_chatbot_service)
    history = chatbot_service.get_past_conversation_messages(conversation_id)

    return history.model_dump()
