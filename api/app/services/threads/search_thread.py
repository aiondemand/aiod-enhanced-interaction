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
from app.schemas.search_results import SearchResults
from app.services.aiod import check_aiod_asset
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

            user_query: BaseUserQuery | None = database.find_by_id(type=query_type, id=query_id)
            if user_query is None:
                err_msg = f"UserQuery id={query_id} doesn't exist even though it should."
                logging.error(err_msg)
                continue

            user_query.update_status(QueryStatus.IN_PROGRESS)
            database.upsert(user_query)

            try:
                results = retrieve_topk_assets_wrapper(
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


# TODO split the function into smaller pieces
def retrieve_topk_assets_wrapper(
    model: AiModel | None,
    llm_query_parser: UserQueryParsing | None,
    embedding_store: EmbeddingStore,
    user_query: BaseUserQuery,
    num_search_retries: int = 5,
) -> SearchResults:
    asset_ids_to_exclude_from_search: list[int] = []
    asset_ids_to_remove_from_db: list[int] = []
    assets_to_return = SearchResults()
    num_assets_to_retrieve = user_query.topk
    meta_filter_str: str = ""
    precomputed_embeddings: list[list[float]] | None = None

    # apply metadata filtering
    if llm_query_parser is not None and isinstance(user_query, FilteredUserQuery):
        if user_query.invoke_llm_for_parsing:
            # utilize LLM to automatically extract filters from the user query
            parsed_query = llm_query_parser(user_query.search_query, user_query.asset_type)
            meta_filter_str = parsed_query["filter_str"]
            user_query.filters = parsed_query["filters"]
        else:
            # user manually defined filters
            meta_filter_str = llm_query_parser.translator_func(
                filters=user_query.filters,
                asset_schema=SchemaOperations.get_asset_schema(user_query.asset_type),
            )
    elif isinstance(user_query, RecommenderUserQuery):
        # Ignore the asset itself from the search
        if user_query.asset_type == user_query.output_asset_type:
            asset_ids_to_exclude_from_search.append(user_query.asset_id)

        precomputed_embeddings = get_precomputed_embeddings_for_recommender(
            model, embedding_store, user_query
        )
        if precomputed_embeddings is None:
            return SearchResults()

    # In Recommender it is necessary to compare asset_id with other output assets
    target_asset_type = (
        user_query.output_asset_type
        if isinstance(user_query, RecommenderUserQuery)
        else user_query.asset_type
    )

    for _ in range(num_search_retries):
        filter_str = f"asset_id not in {asset_ids_to_exclude_from_search}"
        if len(meta_filter_str) > 0:
            filter_str = f"({meta_filter_str}) and ({filter_str})"

        search_query = getattr(user_query, "search_query", "")
        results = embedding_store.retrieve_topk_asset_ids(
            model=model,
            query_text=search_query,
            asset_type=target_asset_type,
            topk=num_assets_to_retrieve,
            filter=filter_str,
            query_embeddings=precomputed_embeddings,
        )

        asset_ids_to_exclude_from_search.extend(results.asset_ids)
        if len(results) == 0:
            break

        # check what assets are still valid
        exists_mask = np.array(
            [
                check_aiod_asset(
                    asset_id,
                    target_asset_type,
                    sleep_time=settings.AIOD.SEARCH_WAIT_INBETWEEN_REQUESTS_SEC,
                )
                for asset_id in results.asset_ids
            ]
        )
        asset_ids_to_del = [results.asset_ids[idx] for idx in np.where(~exists_mask)[0]]
        asset_ids_to_remove_from_db.extend(asset_ids_to_del)

        # perform another Milvus extraction if we dont have sufficient amount of assets as a response
        assets_to_return += results.filter_out_assets(asset_ids_to_del)
        num_assets_to_retrieve = user_query.topk - len(assets_to_return.asset_ids)
        if num_assets_to_retrieve == 0:
            break

    # delete invalid assets from Milvus => lazy delete
    if len(asset_ids_to_remove_from_db) > 0:
        embedding_store.remove_embeddings(asset_ids_to_remove_from_db, target_asset_type)
        logging.info(
            f"[LAZY DELETE] {len(asset_ids_to_remove_from_db)} assets ({target_asset_type.value}) have been deleted"
        )

    return assets_to_return
