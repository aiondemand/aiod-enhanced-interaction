from __future__ import annotations

import logging
import os
from queue import Queue

import numpy as np
from app.config import settings
from app.models.query import UserQuery
from app.schemas.enums import QueryStatus
from app.schemas.search_results import SearchResults
from app.services.database import Database
from app.services.embedding_store import EmbeddingStore, Milvus_EmbeddingStore
from app.services.inference.model import AiModel
from app.services.threads.delete_thread import check_document_existence
from tinydb import Query

QUERY_QUEUE = Queue()


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

    logging.info(
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
            logging.info(f"Searching relevant assets for query ID: {query_id}")

            user_query = database.queries.find_by_id(query_id)
            if user_query is None:
                err_msg = (
                    f"UserQuery id={query_id} doesn't exist even though it should."
                )
                logging.error(err_msg)
                continue

            user_query.update_status(QueryStatus.IN_PROGESS)
            database.queries.upsert(user_query)

            results = retrieve_topk_documents_wrapper(
                model, embedding_store, user_query
            )
            user_query.result_set = results
            user_query.update_status(QueryStatus.COMPLETED)
            database.queries.upsert(user_query)
    except Exception:
        logging.error(
            "An error has been encountered in the query processing thread. "
            + "Entire Application is being terminated now"
        )
        os._exit(1)


def retrieve_topk_documents_wrapper(
    model: AiModel,
    embedding_store: EmbeddingStore,
    user_query: UserQuery,
    num_search_retries: int = 5,
) -> SearchResults:
    collection_name = settings.MILVUS.get_collection_name(user_query.asset_type)

    doc_ids_to_exclude_from_search = []
    doc_ids_to_remove_from_db = []
    documents_to_return = SearchResults()
    num_docs_to_retrieve = user_query.topk

    for _ in range(num_search_retries):
        filter_str = (
            f"doc_id not in {doc_ids_to_exclude_from_search}"
            if len(doc_ids_to_exclude_from_search)
            else ""
        )

        results = embedding_store.retrieve_topk_document_ids(
            model,
            user_query.query,
            collection_name=collection_name,
            topk=num_docs_to_retrieve,
            filter=filter_str,
        )
        doc_ids_to_exclude_from_search.extend(results.doc_ids)

        # check what documents are still valid
        exists_mask = np.array(
            [
                check_document_existence(
                    doc_id,
                    user_query.asset_type,
                    sleep_time=settings.AIOD.SEARCH_WAIT_INBETWEEN_REQUESTS_SEC,
                )
                for doc_id in results.doc_ids
            ]
        )
        doc_ids_to_del = [results.doc_ids[idx] for idx in np.where(~exists_mask)[0]]
        doc_ids_to_remove_from_db.extend(doc_ids_to_del)

        # perform another Milvus extraction if we dont have sufficient amount of
        # documents as a response
        documents_to_return += results.filter_out_docs(doc_ids_to_del)
        num_docs_to_retrieve = user_query.topk - len(documents_to_return.doc_ids)
        if num_docs_to_retrieve == 0:
            break

    # delete invalid documents from Milvus => lazy delete
    if len(doc_ids_to_remove_from_db) > 0:
        embedding_store.remove_embeddings(doc_ids_to_remove_from_db, collection_name)
        logging.info(
            f"[LAZY DELETE] {len(doc_ids_to_remove_from_db)} assets ({user_query.asset_type.value}) have been deleted"
        )
    return documents_to_return
