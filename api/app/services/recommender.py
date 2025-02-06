import logging
import requests
import torch
from fastapi import HTTPException
from app.schemas.search_results import SearchResults
from app.models.query import SimilarQuery
from app.services.embedding_store import MilvusEmbeddingStore
from app.services.inference.text_operations import ConvertJsonToString
from app.services.inference.model import AiModel
from app.schemas.enums import AssetType
from api.app.config import settings


async def get_similar_query_response(
    asset_id: str, asset_type: AssetType, topk: int
) -> dict:
    embedding_store = await MilvusEmbeddingStore.init()
    collection_name = embedding_store.get_collection_name(asset_type)
    embeddings = embedding_store.get_embeddings(asset_id, collection_name)

    if not embeddings:
        return await _fallback_fetch_and_search(
            asset_id, asset_type, topk, embedding_store
        )

    try:
        search_results = embedding_store.retrieve_topk_document_ids(
            model=None,
            asset_type=asset_type,
            topk=topk,
            doc_id=asset_id,
            filter="",
        )

        if not search_results.doc_ids:
            return await _fallback_fetch_and_search(
                asset_id, asset_type, topk, embedding_store
            )

        similar_query = SimilarQuery(
            doc_id=asset_id, asset_type=asset_type, topk=topk, result_set=search_results
        )
        return similar_query.map_to_response()

    except ValueError:
        return await _fallback_fetch_and_search(
            asset_id, asset_type, topk, embedding_store
        )


async def _fallback_fetch_and_search(
    asset_id: str,
    asset_type: AssetType,
    topk: int,
    embedding_store: MilvusEmbeddingStore,
) -> dict:
    logging.info(
        f"No embeddings found in Milvus for '{asset_id}'. Fetching external data..."
    )

    dataset_info = _fetch_external_data(asset_id, asset_type)

    text_data = ConvertJsonToString.stringify(dataset_info)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and settings.USE_GPU else "cpu"
    )
    model = AiModel(device=device)
    tensor = model.compute_asset_embeddings([text_data])
    embeddings_list = [emb.cpu().numpy() for emb in tensor]

    for emb in embeddings_list:
        embedding_store.store_embedding(
            doc_id=asset_id, asset_type=asset_type, embedding=emb
        )

    search_results = embedding_store.retrieve_topk_document_ids(
        model=None, asset_type=asset_type, topk=topk, doc_id=asset_id, filter=""
    )
    if not search_results.doc_ids:
        raise HTTPException(
            status_code=404,
            detail=f"No similar assets found even after embedding '{asset_id}'.",
        )

    similar_query = SimilarQuery(
        search_query=asset_id,
        asset_type=asset_type,
        topk=topk,
        result_set=search_results,
    )
    return similar_query.map_to_response()


def _fetch_external_data(asset_id: str, asset_type: AssetType) -> dict:
    url = f"{settings.AIOD.URL}{asset_type.value}/v1/{asset_id}"
    response = requests.get(url, params={"schema": "aiod"})
    if response.status_code != 200:
        raise HTTPException(
            status_code=404,
            detail=f"External API request failed for asset_id='{asset_id}', asset_type='{asset_type.value}'.",
        )
    return response.json()
