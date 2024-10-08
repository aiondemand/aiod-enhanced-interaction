from __future__ import annotations

import logging
import os
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


async def search_thread() -> None:
    try:
        model = AiModel("cpu")
        embedding_store = await Milvus_EmbeddingStore.init()

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
                err_msg = (
                    f"UserQuery id={query_id} doesn't exist even though it should."
                )
                logger.error(err_msg)
                continue

            userQuery.update_status(QueryStatus.IN_PROGESS)
            database.queries.upsert(userQuery)
            collection_name = settings.MILVUS.get_collection_name(userQuery.asset_type)

            results = embedding_store.retrieve_topk_document_ids(
                model,
                userQuery.query,
                collection_name=collection_name,
                topk=userQuery.topk,
            )
            userQuery.result_set = results
            userQuery.update_status(QueryStatus.COMPLETED)
            database.queries.upsert(userQuery)
    except Exception:
        logger.error(
            "An error has been encountered in the query processing thread. "
            + "Entire Application is being terminated now"
        )
        os._exit(1)
