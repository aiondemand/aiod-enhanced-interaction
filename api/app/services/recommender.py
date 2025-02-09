import logging

import requests
import torch
from app.config import settings
from app.models.query import SimilarQuery
from app.schemas.enums import AssetType
from app.schemas.search_results import SearchResults
from app.services.embedding_store import MilvusEmbeddingStore
from app.services.inference.model import AiModel
from app.services.inference.text_operations import ConvertJsonToString
from fastapi import HTTPException


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
        model=None,
        query_text=str,
        doc_id=asset_id,
        asset_type=asset_type,
        topk=topk,
        filter="",
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


def combine_search_results(
    results_list: list[SearchResults], topk: int = None
) -> SearchResults:

    all_docs = []
    for sr in results_list:
        for doc_id, dist in zip(sr.doc_ids, sr.distances):
            all_docs.append({"doc_id": doc_id, "distance": dist})

    doc_map = {}
    for doc in all_docs:
        d_id = doc["doc_id"]
        d_dist = doc["distance"]
        if d_id not in doc_map or d_dist < doc_map[d_id]["distance"]:
            doc_map[d_id] = doc

    sorted_docs = sorted(doc_map.values(), key=lambda x: x["distance"], reverse=False)

    if topk is not None:
        sorted_docs = sorted_docs[:topk]

    final_doc_ids = [doc["doc_id"] for doc in sorted_docs]
    final_distances = [doc["distance"] for doc in sorted_docs]
    return SearchResults(doc_ids=final_doc_ids, distances=final_distances)
