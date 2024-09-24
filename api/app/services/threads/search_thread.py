from __future__ import annotations

import logging
import threading
from queue import Queue

from app.config import settings
from app.models.query import UserQuery
from app.schemas.query_status import QueryStatus
from app.services.database import UserQueryDatabase
from app.services.embedding_store import Milvus_EmbeddingStore
from app.services.inference.model import AiModelForUserQueries
from tinydb import Query

QUERY_QUEUE = Queue()
logger = logging.getLogger("uvicorn")


def fill_query_queue(query_database: UserQueryDatabase) -> None:
    queries_to_process = query_database.search(
        (Query().status == QueryStatus.IN_PROGESS)
        | (Query().status == QueryStatus.QUEUED)
    )
    queries_to_process = sorted(
        queries_to_process, key=UserQuery.sort_function_to_populate_queue
    )
    for query in queries_to_process:
        QUERY_QUEUE.put(query.id)


def search_thread() -> None:
    model = AiModelForUserQueries()

    # already instantiated singletons
    vector_store = Milvus_EmbeddingStore()
    query_database = UserQueryDatabase()

    fill_query_queue(query_database)

    while True:
        query_id = QUERY_QUEUE.get()
        if query_id is None:
            return
        logging.info(f"Searching relevant assets for query ID: {query_id}")

        userQuery = query_database.find_by_id(query_id=query_id)
        userQuery.update_status(QueryStatus.IN_PROGESS)
        query_database.upsert(userQuery)

        results = vector_store.retrieve_topk_document_ids(
            model,
            query_id,
            userQuery.query,
            collection_name=settings.MILVUS.COLLECTION,
            topk=settings.MILVUS.TOPK,
        )
        userQuery.result_set = results
        userQuery.update_status(QueryStatus.COMPLETED)
        query_database.upsert(userQuery)


def start_search_thread() -> threading.Thread:
    thread = threading.Thread(target=search_thread)
    thread.start()
    return thread
