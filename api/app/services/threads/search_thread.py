from __future__ import annotations

import logging
import os
from queue import Queue

import numpy as np
from app.config import settings
from app.models.query import BaseUserQuery, FilteredUserQuery, SimpleUserQuery
from app.schemas.enums import QueryStatus
from app.schemas.search_results import SearchResults
from app.services.database import Database
from app.services.embedding_store import EmbeddingStore, Milvus_EmbeddingStore
from app.services.inference.llm_query_parsing import Prep_LLM, UserQueryParsing
from app.services.inference.model import AiModel
from app.services.threads.delete_thread import check_document_existence
from tinydb import Query

QUERY_QUEUE = Queue()


def fill_query_queue(database: Database) -> None:
    simple_queries_to_process = database.search(
        SimpleUserQuery,
        (Query().status == QueryStatus.IN_PROGESS)
        | (Query().status == QueryStatus.QUEUED),
    )
    filtered_queries_to_process = database.search(
        FilteredUserQuery,
        (Query().status == QueryStatus.IN_PROGESS)
        | (Query().status == QueryStatus.QUEUED),
    )

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
        model = AiModel("cpu")
        embedding_store = await Milvus_EmbeddingStore.init()

        llm_query_parser = None
        if settings.PERFORM_LLM_QUERY_PARSING:
            llm_query_parser = UserQueryParsing(
                llm=Prep_LLM.setup_ollama_llm(ollama_uri=str(settings.OLLAMA.URI))
            )

        # Singleton - already instantialized
        database = Database()
        fill_query_queue(database)

        while True:
            query_id, query_type = QUERY_QUEUE.get()
            if query_id is None:
                return
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
    num_docs_to_retrieve = user_query.topk

    topic = user_query.orig_query
    meta_filter_str = ""

    if llm_query_parser is None and isinstance(user_query, FilteredUserQuery):
        # TODO deal with this properly
        raise ValueError("We currently do not support filtered queries")

    # apply metadata filtering
    if llm_query_parser is not None and isinstance(user_query, FilteredUserQuery):
        if user_query.invoke_llm_for_parsing:
            # utilize LLM to automatically extract filters from the user query
            parsed_query = llm_query_parser(
                user_query.orig_query, user_query.asset_type
            )
            topic, meta_filter_str = parsed_query["topic"], parsed_query["filter_str"]
            user_query.update_query_metadata(topic, parsed_query["conditions"])
        else:
            # user manually defined filters
            meta_filter_str = llm_query_parser.translator_func(
                conditions=[f.model_dump() for f in user_query.filters],
                asset_schema=llm_query_parser.get_asset_schema(user_query.asset_type),
            )

    for _ in range(num_search_retries):
        filter_str = f"doc_id not in {doc_ids_to_exclude_from_search}"
        if len(meta_filter_str) > 0:
            filter_str = f"({meta_filter_str}) and ({filter_str})"

        results = embedding_store.retrieve_topk_document_ids(
            model,
            topic,
            asset_type=user_query.asset_type,
            topk=num_docs_to_retrieve,
            filter=filter_str,
        )
        if len(results) == 0:
            return documents_to_return

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
        embedding_store.remove_embeddings(
            doc_ids_to_remove_from_db, user_query.asset_type
        )
        logging.info(
            f"[LAZY DELETE] {len(doc_ids_to_remove_from_db)} assets ({user_query.asset_type.value}) have been deleted"
        )
    return documents_to_return
