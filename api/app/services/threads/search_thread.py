from __future__ import annotations

import logging
import os
from queue import Queue
from typing import Type

import numpy as np
from app.config import settings
from app.models.query import (
    BaseUserQuery,
    FilteredUserQuery,
    RecommenderUserQuery,
    SimpleUserQuery,
)
from app.schemas.asset_metadata.base import SchemaOperations
from app.schemas.enums import QueryStatus
from app.schemas.params import MilvusSearchParams
from app.schemas.search_results import SearchResults
from app.services.aiod import check_aiod_document
from app.services.database import Database
from app.services.embedding_store import EmbeddingStore, MilvusEmbeddingStore
from app.services.inference.llm_query_parsing import PrepareLLM, UserQueryParsing
from app.services.inference.model import AiModel
from app.services.recommender import get_precomputed_embeddings_for_recommender
from tinydb import Query

QUERY_QUEUE: Queue[tuple[str | None, Type[BaseUserQuery] | None]] = Queue()


def fill_query_queue(database: Database) -> None:
    condition = (Query().status == QueryStatus.IN_PROGRESS) | (Query().status == QueryStatus.QUEUED)
    simple_queries_to_process = database.search(SimpleUserQuery, condition)
    filtered_queries_to_process = database.search(FilteredUserQuery, condition)
    similar_queries_to_process = database.search(RecommenderUserQuery, condition)

    if (
        len(simple_queries_to_process + filtered_queries_to_process + similar_queries_to_process)
        == 0
    ):
        return

    queries_to_process = sorted(
        simple_queries_to_process + filtered_queries_to_process + similar_queries_to_process,
        key=BaseUserQuery.sort_function_to_populate_queue,
    )
    for query in queries_to_process:
        QUERY_QUEUE.put((query.id, type(query)))

    logging.info(
        f"Query queue has been populated with {len(queries_to_process)} queries to process."
    )


# TODO make this function less ugly...
async def search_thread() -> None:
    try:
        # Singleton - already instantiated
        database = Database()
        fill_query_queue(database)

        model = AiModel("cpu")
        embedding_store = await MilvusEmbeddingStore.init()

        llm_query_parser = None
        if settings.PERFORM_LLM_QUERY_PARSING:
            llm_query_parser = UserQueryParsing(llm=PrepareLLM.setup_ollama_llm())

        while True:
            query_id, query_type = QUERY_QUEUE.get()
            if query_id is None:
                return
            if query_type == FilteredUserQuery and settings.PERFORM_LLM_QUERY_PARSING is False:
                continue
            logging.info(f"Searching relevant assets for query ID: {query_id}")

            user_query: BaseUserQuery = database.find_by_id(type=query_type, id=query_id)
            if user_query is None:
                err_msg = f"UserQuery id={query_id} doesn't exist even though it should."
                logging.error(err_msg)
                continue

            user_query.update_status(QueryStatus.IN_PROGRESS)
            database.upsert(user_query)

            try:
                results = search_documents_wrapper(
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
                logging.error(f"Error encountered while processing query ID: {query_id}")
    except Exception as e:
        logging.error(e)
        logging.error(
            "The above error has been encountered in the query processing thread. "
            + "Entire Application is being terminated now"
        )
        os._exit(1)


def search_documents_wrapper(
    model: AiModel | None,
    llm_query_parser: UserQueryParsing | None,
    embedding_store: EmbeddingStore,
    user_query: BaseUserQuery,
    num_search_retries: int = 5,
) -> SearchResults:
    doc_ids_to_remove_from_db: list[str] = []
    all_documents_to_return = SearchResults()

    search_params = prepare_search_parameters(model, llm_query_parser, embedding_store, user_query)
    if search_params is None:
        return all_documents_to_return

    for _ in range(num_search_retries):
        new_results = embedding_store.retrieve_topk_document_ids(search_params)
        all_documents_to_return = all_documents_to_return + validate_documents(
            new_results, search_params, doc_ids_to_remove_from_db
        )

        search_params.topk = user_query.topk - len(all_documents_to_return)
        if search_params.topk == 0 or len(new_results) == 0:
            break

    # delete invalid documents from Milvus => lazy delete
    if len(doc_ids_to_remove_from_db) > 0:
        embedding_store.remove_embeddings(doc_ids_to_remove_from_db, search_params.asset_type)
        logging.info(
            f"[LAZY DELETE] {len(doc_ids_to_remove_from_db)} assets ({search_params.asset_type.value}) have been deleted"
        )

    return all_documents_to_return


def prepare_search_parameters(
    model: AiModel | None,
    llm_query_parser: UserQueryParsing | None,
    embedding_store: EmbeddingStore,
    user_query: BaseUserQuery,
) -> MilvusSearchParams | None:
    # apply metadata filtering
    metadata_filter_str = ""
    if llm_query_parser is not None and isinstance(user_query, FilteredUserQuery):
        if user_query.invoke_llm_for_parsing:
            # utilize LLM to automatically extract filters from the user query
            parsed_query = llm_query_parser(user_query.search_query, user_query.asset_type)
            metadata_filter_str = parsed_query["filter_str"]
            user_query.filters = parsed_query["filters"]
        else:
            # user manually defined filters
            metadata_filter_str = llm_query_parser.translator_func(
                filters=user_query.filters,
                asset_schema=SchemaOperations.get_asset_schema(user_query.asset_type),
            )

    # compute query embedding
    if isinstance(user_query, RecommenderUserQuery):
        query_embeddings = get_precomputed_embeddings_for_recommender(
            model, embedding_store, user_query
        )
        if query_embeddings is None:
            return None
    else:
        query_embeddings = model.compute_query_embeddings(user_query.search_query)

    # select asset type to search for
    target_asset_type = (
        user_query.output_asset_type
        if isinstance(user_query, RecommenderUserQuery)
        else user_query.asset_type
    )
    # ignore the asset itself from the search if necessary
    doc_ids_to_exclude_from_search = (
        str(user_query.asset_id)
        if isinstance(user_query, RecommenderUserQuery)
        and user_query.asset_type == user_query.output_asset_type
        else []
    )

    return MilvusSearchParams(
        data=query_embeddings,
        topk=user_query.topk,
        asset_type=target_asset_type,
        metadata_filter=metadata_filter_str,
        doc_ids_to_exclude=doc_ids_to_exclude_from_search,
    )


def validate_documents(
    results: SearchResults,
    search_params: MilvusSearchParams,
    doc_ids_to_remove_from_db: list[str],
) -> SearchResults:
    if len(results) == 0:
        return results

    # check what documents are still valid
    exists_mask = np.array(
        [
            check_aiod_document(
                doc_id,
                search_params.asset_type,
                sleep_time=settings.AIOD.SEARCH_WAIT_INBETWEEN_REQUESTS_SEC,
            )
            for doc_id in results.doc_ids
        ]
    )

    doc_ids_to_del = [results.doc_ids[idx] for idx in np.where(~exists_mask)[0]]
    search_params.doc_ids_to_exclude.extend(results.doc_ids)
    doc_ids_to_remove_from_db.extend(doc_ids_to_del)

    return results.filter_out_docs_by_ids(doc_ids_to_del)
