from __future__ import annotations

from copy import deepcopy
import logging
from time import sleep
from typing import Type

from uuid import UUID
import numpy as np
from app import settings
from app.models.query import BaseUserQuery
from app.models import FilteredUserQuery, RecommenderUserQuery, SimpleUserQuery
from app.schemas.asset_id import AssetId
from app.schemas.enums import QueryStatus, SupportedAssetType
from app.schemas.params import VectorSearchParams
from app.schemas.search_results import AssetResults, SearchResults
from app.services.aiod import get_aiod_asset
from app.services.embedding_store import EmbeddingStore
from app.services.metadata_filtering.query_parsing_agent import QueryParsingWrapper
from app.services.inference.model import AiModel
from app.services.resilience import LocalServiceUnavailableException
from app.services.inference.text_operations import ConvertJsonToString


async def semantic_search_wrapper(
    query_id: UUID,
    query_type: Type[BaseUserQuery],
    model: AiModel,
    embedding_store: EmbeddingStore,
) -> dict:
    user_query = await fetch_user_query(query_id, query_type)
    if user_query is None:
        return {"status": "skipped", "reason": "Query not found or invalid"}

    logging.info(f"Searching relevant assets for query ID: {str(query_id)}")

    try:
        results = await search_across_assets_wrapper(model, embedding_store, user_query)
        user_query.result_set = results
        user_query.update_status(QueryStatus.COMPLETED)
        await user_query.replace_doc()

        return {
            "status": "completed",
            "query_id": str(query_id),
            "results_count": len(results.asset_ids) if results else 0,
        }
    except LocalServiceUnavailableException as e:
        logging.error(e)
        logging.error(
            "The above error has been encountered in the query processing task. "
            + "Task will be retried."
        )
        # Update query status to failed
        user_query.update_status(QueryStatus.FAILED)
        await user_query.replace_doc()
        # Re-raise to trigger Celery retry mechanism
        raise
    except Exception as e:
        user_query.update_status(QueryStatus.FAILED)
        await user_query.replace_doc()
        logging.error(e)
        logging.error(
            f"The above error has been encountered in the query processing task while processing query ID: {str(query_id)}"
        )
        return {"status": "failed", "error": str(e)}


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


async def search_across_assets_wrapper(
    model: AiModel,
    embedding_store: EmbeddingStore,
    user_query: BaseUserQuery,
) -> AssetResults:
    search_params, all_asset_types = await prepare_search_parameters(
        model, embedding_store, user_query
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
        # artificial halt to decrease the load on Metadata Catalogue that caused some issues prior
        sleep(1)

    return AssetResults.merge_results(search_results, k=user_query.topk)


def search_asset_collection(
    embedding_store: EmbeddingStore,
    search_params: VectorSearchParams,
    user_query: BaseUserQuery,
    num_search_retries: int = 5,
) -> AssetResults:
    asset_ids_to_remove_from_db: list[AssetId] = []
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


async def prepare_search_parameters(
    model: AiModel,
    embedding_store: EmbeddingStore,
    user_query: BaseUserQuery,
) -> tuple[VectorSearchParams | None, bool]:
    # apply metadata filtering
    metadata_filter_str = ""

    if settings.PERFORM_LLM_QUERY_PARSING and isinstance(user_query, FilteredUserQuery):
        if user_query.invoke_llm_for_parsing:
            # utilize LLM to automatically extract filters from the user query
            parsed_query = await QueryParsingWrapper.parse_query(
                user_query.search_query, user_query.asset_type
            )
            metadata_filter_str = parsed_query["filter_str"]
            user_query.filters = parsed_query["filters"]
        elif user_query.filters is not None:
            # user manually defined filters
            metadata_filter_str = QueryParsingWrapper.milvus_translate(
                user_query.filters, user_query.asset_type
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
    asset_ids_to_exclude_from_search: list[AssetId] = (
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
    asset_ids_to_remove_from_db: list[AssetId],
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


def get_precomputed_embeddings_for_recommender(
    model: AiModel,
    embedding_store: EmbeddingStore,
    user_query: RecommenderUserQuery,
) -> list[list[float]] | None:
    precomputed_embeddings = embedding_store.get_asset_embeddings(
        user_query.asset_id, user_query.asset_type
    )

    # artificial halt to decrease the load on Metadata Catalogue that caused some issues prior
    sleep(1)

    if precomputed_embeddings is None:
        logging.warning(
            f"No embedding found for asset_id='{user_query.asset_id}' ({user_query.asset_type.value}) in Milvus."
        )
        asset_obj = get_aiod_asset(user_query.asset_id, user_query.asset_type)
        if asset_obj is None:
            # TODO we should pass the information to the user that the asset_id they provided is invalid
            # For now, current implementation returns an empty list of similar assets to a non-existing asset
            logging.error(
                f"Asset with id '{user_query.asset_id}' ({user_query.asset_type.value}) not found in AIoD platform."
            )
            return None
        stringified_asset = ConvertJsonToString.stringify(asset_obj)
        emb = model.compute_asset_embeddings(stringified_asset)[0]
        precomputed_embeddings = emb.cpu().numpy().tolist()
    return precomputed_embeddings
