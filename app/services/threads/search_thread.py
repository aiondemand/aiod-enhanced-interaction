from __future__ import annotations

from copy import deepcopy
import logging
import os
from queue import Queue
from typing import Type

from uuid import UUID
from beanie.odm.operators.find.logical import Or
import numpy as np
from app.config import settings
from app.models.query import (
    BaseUserQuery,
    FilteredUserQuery,
    RecommenderUserQuery,
    SimpleUserQuery,
)
from app.schemas.asset_metadata.operations import SchemaOperations
from app.schemas.enums import QueryStatus, SupportedAssetType
from app.schemas.params import VectorSearchParams
from app.schemas.search_results import AssetResults, SearchResults
from app.services.aiod import get_aiod_asset
from app.services.embedding_store import EmbeddingStore, MilvusEmbeddingStore
from app.services.metadata_filtering.llm_query_parsing import PrepareLLM, UserQueryParsing
from app.services.inference.model import AiModel
from app.services.recommender import get_precomputed_embeddings_for_recommender
from app.services.resilience import LocalServiceUnavailableException


QUERY_QUEUE: Queue[tuple[UUID | None, Type[BaseUserQuery] | None]] = Queue()


async def fill_query_queue() -> None:
    async def __retrieve_queries(typ: type[BaseUserQuery]) -> list[BaseUserQuery]:
        return await typ.find_all_docs(
            Or(typ.status == QueryStatus.QUEUED, typ.status == QueryStatus.IN_PROGRESS)
        )

    simple_queries_to_process: list[BaseUserQuery] = await __retrieve_queries(SimpleUserQuery)
    filtered_queries_to_process: list[BaseUserQuery] = await __retrieve_queries(FilteredUserQuery)
    similar_queries_to_process: list[BaseUserQuery] = await __retrieve_queries(RecommenderUserQuery)

    queries_to_process: list[BaseUserQuery] = sorted(
        simple_queries_to_process + filtered_queries_to_process + similar_queries_to_process,
        key=BaseUserQuery.sort_function_to_populate_queue,
    )
    if len(queries_to_process) == 0:
        return
    for query in queries_to_process:
        QUERY_QUEUE.put((query.id, type(query)))

    logging.info(
        f"Query queue has been populated with {len(queries_to_process)} queries to process."
    )


async def search_thread() -> None:
    await fill_query_queue()

    model = AiModel("cpu")
    embedding_store = MilvusEmbeddingStore()

    llm_query_parser = None
    if settings.PERFORM_LLM_QUERY_PARSING:
        llm_query_parser = UserQueryParsing(llm=PrepareLLM.setup_ollama_llm())

    while True:
        query_id, query_type = QUERY_QUEUE.get()
        if query_id is None or query_type is None:
            break

        user_query = await fetch_user_query(query_id, query_type)
        if user_query is None:
            continue
        logging.info(f"Searching relevant assets for query ID: {str(query_id)}")

        try:
            results = search_across_assets_wrapper(
                model,
                llm_query_parser,
                embedding_store,
                user_query,
            )
            user_query.result_set = results
            user_query.update_status(QueryStatus.COMPLETED)
            await user_query.replace_doc()
        except LocalServiceUnavailableException as e:
            logging.error(e)
            logging.error(
                "The above error has been encountered in the embedding thread. "
                + "Entire Application is being terminated now"
            )
            os._exit(1)
        except Exception as e:
            user_query.update_status(QueryStatus.FAILED)
            await user_query.replace_doc()
            logging.error(e)
            logging.error(
                f"The above error has been encountered in the query processing thread while processing query ID: {str(query_id)}"
            )


async def fetch_user_query(query_id: UUID, query_type: Type[BaseUserQuery]) -> BaseUserQuery | None:
    if query_type == FilteredUserQuery and settings.PERFORM_LLM_QUERY_PARSING is False:
        return None

    user_query: BaseUserQuery | None = await query_type.get(query_id)
    if user_query is None:
        err_msg = f"UserQuery id={query_id} doesn't exist even though it should."
        logging.error(err_msg)
        return None
    else:
        user_query.update_status(QueryStatus.IN_PROGRESS)
        await user_query.replace_doc()

        return user_query


def search_across_assets_wrapper(
    model: AiModel,
    llm_query_parser: UserQueryParsing | None,
    embedding_store: EmbeddingStore,
    user_query: BaseUserQuery,
) -> AssetResults:
    search_params, all_asset_types = prepare_search_parameters(
        model, llm_query_parser, embedding_store, user_query
    )
    if search_params is None:
        return AssetResults()
    asset_type_list: list[SupportedAssetType] = (
        [search_params.asset_type] if all_asset_types is False else settings.AIOD.ASSET_TYPES
    )

    # perform multiple semantic searches in separate asset_type collections if
    # the user has requested to go over ALL the asset types
    search_results: list[AssetResults] = []
    for asset_type in asset_type_list:
        temp_search_params = deepcopy(search_params)
        temp_search_params.asset_type = asset_type
        search_results.append(
            search_asset_collection(embedding_store, temp_search_params, user_query)
        )

    return AssetResults.merge_results(search_results, k=user_query.topk)


def search_asset_collection(
    embedding_store: EmbeddingStore,
    search_params: VectorSearchParams,
    user_query: BaseUserQuery,
    num_search_retries: int = 5,
) -> AssetResults:
    asset_ids_to_remove_from_db: list[str] = []
    all_assets_to_return = AssetResults()

    for _ in range(num_search_retries):
        new_results = embedding_store.retrieve_topk_asset_ids(search_params)
        validated_new_results = validate_assets(
            new_results, search_params, asset_ids_to_remove_from_db
        ).filter_out_assets_by_id(all_assets_to_return.asset_ids)

        all_assets_to_return = all_assets_to_return + validated_new_results
        search_params.topk = user_query.topk - len(all_assets_to_return)
        if search_params.topk == 0 or len(new_results) == 0:
            break

    # delete invalid assets from Milvus => lazy delete
    if len(asset_ids_to_remove_from_db) > 0:
        embedding_store.remove_embeddings(asset_ids_to_remove_from_db, search_params.asset_type)
        logging.info(
            f"[LAZY DELETE] {len(asset_ids_to_remove_from_db)} assets ({search_params.asset_type.value}) have been deleted"
        )

    return all_assets_to_return


def prepare_search_parameters(
    model: AiModel,
    llm_query_parser: UserQueryParsing | None,
    embedding_store: EmbeddingStore,
    user_query: BaseUserQuery,
) -> tuple[VectorSearchParams | None, bool]:
    # apply metadata filtering
    metadata_filter_str = ""
    if llm_query_parser is not None and isinstance(user_query, FilteredUserQuery):
        if user_query.invoke_llm_for_parsing:
            # utilize LLM to automatically extract filters from the user query
            parsed_query = llm_query_parser(user_query.search_query, user_query.asset_type)
            metadata_filter_str = parsed_query["filter_str"]
            user_query.filters = parsed_query["filters"]
        elif user_query.filters is not None:
            # user manually defined filters
            metadata_filter_str = llm_query_parser.translator_func(
                user_query.filters,
                SchemaOperations.get_asset_schema(user_query.asset_type),
            )

    # compute query embedding
    if isinstance(user_query, RecommenderUserQuery):
        query_embeddings = get_precomputed_embeddings_for_recommender(
            model, embedding_store, user_query
        )
        if query_embeddings is None:
            return None, False
    elif isinstance(user_query, (SimpleUserQuery, FilteredUserQuery)):
        query_embeddings = model.compute_query_embeddings(user_query.search_query)
    else:
        raise ValueError("We don't support other types of user queries yet")

    # select asset type to search for
    target_asset_type_temp = (
        user_query.output_asset_type
        if isinstance(user_query, RecommenderUserQuery)
        else user_query.asset_type
    )
    all_asset_types = target_asset_type_temp.is_all()
    target_asset_type = (
        target_asset_type_temp.to_SupportedAssetType()
        if all_asset_types is False
        else SupportedAssetType.DATASETS  # a placeholder value that will be replaced
    )
    # ignore the asset itself from the search if necessary
    asset_ids_to_exclude_from_search = (
        [user_query.asset_id]
        if isinstance(user_query, RecommenderUserQuery)
        and user_query.asset_type == user_query.output_asset_type
        else []
    )

    search_params = embedding_store.create_search_params(
        data=query_embeddings,
        topk=user_query.topk,
        asset_type=target_asset_type,
        metadata_filter=metadata_filter_str,
        asset_ids_to_exclude=asset_ids_to_exclude_from_search,
    )
    return search_params, all_asset_types


def validate_assets(
    results: SearchResults,
    search_params: VectorSearchParams,
    asset_ids_to_remove_from_db: list[str],
) -> AssetResults:
    if len(results) == 0:
        return AssetResults()

    # retrieve assets from AIoD Catalogue
    assets = [
        get_aiod_asset(
            asset_id,
            search_params.asset_type,
            sleep_time=settings.AIOD.SEARCH_WAIT_INBETWEEN_REQUESTS_SEC,
        )
        for asset_id in results.asset_ids
    ]
    exists_mask = np.array([asset is not None for asset in assets])
    idx_to_keep = np.where(exists_mask)[0]
    ids_to_del = [results.asset_ids[idx] for idx in np.where(~exists_mask)[0]]

    search_params.asset_ids_to_exclude.extend(results.asset_ids)
    asset_ids_to_remove_from_db.extend(ids_to_del)

    return AssetResults(
        asset_ids=[results.asset_ids[idx] for idx in idx_to_keep],
        distances=[results.distances[idx] for idx in idx_to_keep],
        asset_types=[results.asset_types[idx] for idx in idx_to_keep],
        assets=[assets[idx] for idx in idx_to_keep],
    )
