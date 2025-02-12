from __future__ import annotations

import logging
import os
from queue import Queue
from typing import Type

import numpy as np
import torch
from app.config import settings
from app.models.query import (
    BaseUserQuery,
    FilteredUserQuery,
    SimilarUserQuery,
    SimpleUserQuery,
)
from app.schemas.asset_metadata.base import SchemaOperations
from app.schemas.enums import QueryStatus
from app.schemas.search_results import SearchResults
from app.services import recommender
from app.services.aiod import check_aiod_document, get_aiod_document
from app.services.database import Database
from app.services.embedding_store import EmbeddingStore, MilvusEmbeddingStore
from app.services.inference.llm_query_parsing import PrepareLLM, UserQueryParsing
from app.services.inference.model import AiModel
from app.services.inference.text_operations import ConvertJsonToString
from tinydb import Query

QUERY_QUEUE: Queue[tuple[str | None, Type[BaseUserQuery] | None]] = Queue()


def fill_query_queue(database: Database) -> None:
    condition = (Query().status == QueryStatus.IN_PROGRESS) | (
        Query().status == QueryStatus.QUEUED
    )
    simple_queries_to_process = database.search(SimpleUserQuery, condition)
    filtered_queries_to_process = database.search(FilteredUserQuery, condition)
    similar_queries_to_process = database.search(SimilarUserQuery, condition)

    if (
        len(
            simple_queries_to_process
            + filtered_queries_to_process
            + similar_queries_to_process
        )
        == 0
    ):
        return

    queries_to_process = sorted(
        simple_queries_to_process
        + filtered_queries_to_process
        + similar_queries_to_process,
        key=BaseUserQuery.sort_function_to_populate_queue,
    )
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
        embedding_store = await MilvusEmbeddingStore.init()

        llm_query_parser = None
        if settings.PERFORM_LLM_QUERY_PARSING:
            llm_query_parser = UserQueryParsing(llm=PrepareLLM.setup_ollama_llm())

        while True:
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

            user_query.update_status(QueryStatus.IN_PROGRESS)
            database.upsert(user_query)

            try:
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
    except Exception as e:
        logging.error(e)
        logging.error(
            "The above error has been encountered in the query processing thread. "
            + "Entire Application is being terminated now"
        )
        os._exit(1)


def retrieve_topk_documents_wrapper(
    model: AiModel | None,
    llm_query_parser: UserQueryParsing | None,
    embedding_store: EmbeddingStore,
    user_query: BaseUserQuery,
    num_search_retries: int = 5,
) -> SearchResults:
    doc_ids_to_exclude_from_search = []
    doc_ids_to_remove_from_db = []
    documents_to_return = SearchResults()
    num_docs_to_retrieve = user_query.topk
    meta_filter_str = ""
    precomputed_embedding = None
    precomputed_embeddings_list = None

    # apply metadata filtering
    if llm_query_parser is not None and isinstance(user_query, FilteredUserQuery):
        if user_query.invoke_llm_for_parsing:
            # utilize LLM to automatically extract filters from the user query
            parsed_query = llm_query_parser(
                user_query.search_query, user_query.asset_type
            )
            meta_filter_str = parsed_query["filter_str"]
            user_query.filters = parsed_query["filters"]
        else:
            # user manually defined filters
            meta_filter_str = llm_query_parser.translator_func(
                filters=user_query.filters,
                asset_schema=SchemaOperations.get_asset_schema(user_query.asset_type),
            )
    elif isinstance(user_query, SimilarUserQuery):
        embeddings = embedding_store.get_asset_embeddings(
            user_query.asset_id, user_query.asset_type
        )

        if embeddings:
            precomputed_embedding = embeddings[0]
        else:
            # if the asset is not found in Milvus --> fetch data from AIoD platform
            logging.warning(
                f"No embedding found for doc_id='{user_query.asset_id}' in Milvus."
            )

            dataset_info = get_aiod_document(user_query.asset_id, user_query.asset_type)
            text_data = ConvertJsonToString.stringify(dataset_info)
            device = torch.device(
                "cuda" if torch.cuda.is_available() and settings.USE_GPU else "cpu"
            )
            model = AiModel(device=device)

            tensor = model.compute_asset_embeddings([text_data])
            precomputed_embeddings_list = [emb.cpu().numpy() for emb in tensor]

    for _ in range(num_search_retries):
        filter_str = f"doc_id not in {doc_ids_to_exclude_from_search}"
        if len(meta_filter_str) > 0:
            filter_str = f"({meta_filter_str}) and ({filter_str})"

        # if there are multiple lists of embeddings
        if precomputed_embeddings_list is not None:
            all_results = []
            for emb_array in precomputed_embeddings_list:
                for precomputed_embedding in emb_array:
                    print(precomputed_embedding)
                    candidate_results = embedding_store.retrieve_topk_document_ids(
                        model=None,
                        query_text=None,
                        asset_type=user_query.asset_type,
                        topk=user_query.topk,
                        filter="",
                        precomputed_embedding=precomputed_embedding,
                    )
                    all_results.append(candidate_results)

            #
            results = recommender.combine_search_results(
                all_results, topk=user_query.topk
            )
        else:
            search_query = getattr(user_query, "search_query", "")
            results = embedding_store.retrieve_topk_document_ids(
                model=model,
                query_text=search_query,
                asset_type=user_query.asset_type,
                topk=num_docs_to_retrieve,
                filter=filter_str,
                precomputed_embedding=precomputed_embedding,
            )

        doc_ids_to_exclude_from_search.extend(results.doc_ids)
        if len(results) == 0:
            break

        # check what documents are still valid
        exists_mask = np.array(
            [
                check_aiod_document(
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
