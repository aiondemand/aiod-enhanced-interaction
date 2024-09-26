from __future__ import annotations

import logging
import threading
from queue import Queue

from app.config import settings
from app.models.query import UserQuery
from app.schemas.enums import QueryStatus
from app.services.database import Database
from app.services.embedding_store import Milvus_EmbeddingStore
from app.services.inference.model import AiModel
from tinydb import Query

QUERY_QUEUE = Queue()
logger = logging.getLogger("uvicorn")


def fill_query_queue(database: Database) -> None:
    queries_to_process = database.queries.search(
        (Query().status == QueryStatus.IN_PROGESS)
        | (Query().status == QueryStatus.QUEUED)
    )
    if len(queries_to_process) == 0:
        return

    queries_to_process = sorted(
        queries_to_process, key=UserQuery.sort_function_to_populate_queue
    )
    for query in queries_to_process:
        QUERY_QUEUE.put(query.id)

    logger.info(
        f"Query queue has been populated with {len(queries_to_process)} "
        + "queries to process."
    )


def search_thread() -> None:
    model = AiModel("cpu")
    embedding_store = Milvus_EmbeddingStore()

    # Singleton - already instantialized
    database = Database()
    fill_query_queue(database)

    while True:
        query_id = QUERY_QUEUE.get()
        if query_id is None:
            return
        logger.info(f"Searching relevant assets for query ID: {query_id}")

        userQuery = database.queries.find_by_id(query_id)
        if userQuery is None:
            # TODO
            pass

        userQuery.update_status(QueryStatus.IN_PROGESS)
        database.queries.upsert(userQuery)
        collection_name = settings.MILVUS.get_collection_name(userQuery.asset_type)

        results = embedding_store.retrieve_topk_document_ids(
            model,
            userQuery.query,
            collection_name=collection_name,
            topk=settings.MILVUS.TOPK,
        )
        userQuery.result_set = results
        userQuery.update_status(QueryStatus.COMPLETED)
        database.queries.upsert(userQuery)


def start_search_thread() -> threading.Thread:
    thread = threading.Thread(target=search_thread)
    thread.start()
    return thread
