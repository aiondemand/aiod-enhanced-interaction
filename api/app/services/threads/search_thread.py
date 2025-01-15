from __future__ import annotations

import logging
import os
from asyncio import Condition
from queue import Queue
from time import sleep
from typing import Type

import numpy as np
from app.config import settings
from app.helper import _perform_request
from app.models.query import BaseUserQuery, FilteredUserQuery, SimpleUserQuery
from app.schemas.asset_metadata.base import SchemaOperations
from app.schemas.enums import AssetType, QueryStatus
from app.schemas.search_results import SearchResults
from app.services.database import Database
from app.services.embedding_store import EmbeddingStore, Milvus_EmbeddingStore
from app.services.inference.llm_query_parsing import Prep_LLM, UserQueryParsing
from app.services.inference.model import AiModel
from requests.exceptions import HTTPError
from tinydb import Query

QUERY_QUEUE: Queue[tuple[str | None, Type[BaseUserQuery] | None]] = Queue()
QUERY_CONDITIONS: dict[str, Condition] = {}


def fill_query_queue(database: Database) -> None:
    condition = (Query().status == QueryStatus.IN_PROGESS) | (
        Query().status == QueryStatus.QUEUED
    )
    simple_queries_to_process = database.search(SimpleUserQuery, condition)
    filtered_queries_to_process = database.search(FilteredUserQuery, condition)
    if len(simple_queries_to_process + filtered_queries_to_process) == 0:
        return

    queries_to_process = sorted(
        simple_queries_to_process + filtered_queries_to_process,
        key=BaseUserQuery.sort_function_to_populate_queue,
    )

    # Perhaps also put type of the query into QUERY_QUEUE ???
    for query in queries_to_process:
        QUERY_QUEUE.put((query.id, type(query)))

    logging.info(
        f"Query queue has been populated with {len(queries_to_process)} "
        + "queries to process."
    )


async def search_thread() -> None:
    try:
        # Singleton - already instantialized
        database = Database()
        fill_query_queue(database)

        model = AiModel("cpu")
        embedding_store = await Milvus_EmbeddingStore.init()

        llm_query_parser = None
        if settings.PERFORM_LLM_QUERY_PARSING:
            llm_query_parser = UserQueryParsing(llm=Prep_LLM.setup_ollama_llm())

        while True:
            try:
                query_id, query_type = QUERY_QUEUE.get()
                if query_id is None:
                    return
                if (
                    query_type == FilteredUserQuery
                    and settings.PERFORM_LLM_QUERY_PARSING is False
                ):
                    continue

                logging.info(f"Searching relevant assets for query ID: {query_id}")

                user_query: BaseUserQuery = database.find_by_id(
                    type=query_type, id=query_id
                )
                if user_query is None:
                    err_msg = (
                        f"UserQuery id={query_id} doesn't exist even though it should."
                    )
                    logging.error(err_msg)
                    continue

                user_query.update_status(QueryStatus.IN_PROGESS)
                database.upsert(user_query)

                results = retrieve_topk_documents_wrapper(
                    model,
                    llm_query_parser,
                    embedding_store,
                    user_query,
                )
                user_query.result_set = results
                user_query.update_status(QueryStatus.COMPLETED)
                database.upsert(user_query)
            except Exception as e:
                user_query.update_status(QueryStatus.FAILED)
                database.upsert(user_query)
                logging.error(e)
                logging.error(
                    f"Error encountered while processing query ID: {query_id}"
                )
            finally:
                # notify blocking endpoints that the query results have been computed
                if QUERY_CONDITIONS.get(user_query.id, None) is not None:
                    async with QUERY_CONDITIONS[user_query.id]:
                        QUERY_CONDITIONS[user_query.id].notify_all()

    except Exception as e:
        logging.error(e)
        logging.error(
            "The above error has been encountered in the query processing thread. "
            + "Entire Application is being terminated now"
        )
        os._exit(1)


def retrieve_topk_documents_wrapper(
    model: AiModel,
    llm_query_parser: UserQueryParsing | None,
    embedding_store: EmbeddingStore,
    user_query: BaseUserQuery,
    num_search_retries: int = 5,
) -> SearchResults:
    doc_ids_to_exclude_from_search = []
    doc_ids_to_remove_from_db = []
    documents_to_return = SearchResults()
    num_docs_to_retrieve = user_query.limit

    topic = user_query.search_query
    meta_filter_str = ""

    # apply metadata filtering
    if llm_query_parser is not None and isinstance(user_query, FilteredUserQuery):
        if user_query.invoke_llm_for_parsing:
            # utilize LLM to automatically extract filters from the user query
            parsed_query = llm_query_parser(
                user_query.search_query, user_query.asset_type
            )
            meta_filter_str = parsed_query["filter_str"]
            user_query.update_query_metadata(topic, parsed_query["filters"])
        else:
            # user manually defined filters
            meta_filter_str = llm_query_parser.translator_func(
                filters=user_query.filters,
                asset_schema=SchemaOperations.get_asset_schema(user_query.asset_type),
            )

    for _ in range(num_search_retries):
        filter_str = f"doc_id not in {doc_ids_to_exclude_from_search}"
        if len(meta_filter_str) > 0:
            filter_str = f"({meta_filter_str}) and ({filter_str})"

        results = embedding_store.retrieve_topk_document_ids(
            model,
            topic,
            asset_type=user_query.asset_type,
            offset=user_query.offset,
            limit=num_docs_to_retrieve,
            filter=filter_str,
        )
        doc_ids_to_exclude_from_search.extend(results.doc_ids)
        if len(results) == 0:
            break

        results.documents = np.array(
            [
                get_aiod_document(
                    doc_id,
                    user_query.asset_type,
                    sleep_time=settings.AIOD.SEARCH_WAIT_INBETWEEN_REQUESTS_SEC,
                )
                for doc_id in results.doc_ids
            ]
        )
        # check what documents are still valid
        doc_ids_to_del = results.filter_out_docs()
        doc_ids_to_remove_from_db.extend(doc_ids_to_del)

        # perform another Milvus extraction if we dont have sufficient amount of
        # documents as a response
        documents_to_return += results
        num_docs_to_retrieve = user_query.limit - len(documents_to_return.doc_ids)
        if num_docs_to_retrieve == 0:
            break

    # delete invalid documents from Milvus => lazy delete
    if len(doc_ids_to_remove_from_db) > 0:
        embedding_store.remove_embeddings(
            doc_ids_to_remove_from_db, user_query.asset_type
        )
        logging.info(
            f"[LAZY DELETE] {len(doc_ids_to_remove_from_db)} assets ({user_query.asset_type.value}) have been deleted"
        )

    # Compute num_hits in the database that match the query/filters
    documents_to_return.num_hits = embedding_store.get_number_of_hits(
        asset_type=user_query.asset_type, filter=meta_filter_str
    )
    return documents_to_return


def get_aiod_document(
    doc_id: str, asset_type: AssetType, sleep_time: float = 0.1
) -> dict | None:
    try:
        sleep(sleep_time)
        response = _perform_request(
            settings.AIOD.get_asset_by_id_url(doc_id, asset_type)
        )
        return response.json()
    except HTTPError as e:
        if e.response.status_code == 404:
            return None
        raise


def check_aiod_document(
    doc_id: str, asset_type: AssetType, sleep_time: float = 0.1
) -> bool:
    return get_aiod_document(doc_id, asset_type, sleep_time) is not None
