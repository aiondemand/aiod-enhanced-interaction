import logging
import torch
import requests

from fastapi import HTTPException
from app.schemas.search_results import SearchResults
from app.models.query import SimilarQuery
from app.services.embedding_store import Milvus_EmbeddingStore
from app.services.inference.text_operations import ConvertJsonToString
from app.services.inference.model import AiModel
from api.app.config import settings
from app.schemas.enums import AssetType


async def get_similar_query_response(
    asset_id: str, asset_type: AssetType, topk: int
) -> dict:
    embedding_store = await Milvus_EmbeddingStore.init()
    collection_name = settings.MILVUS.get_collection_name(asset_type)
    embeddings = embedding_store.get_embeddings(asset_id, collection_name)

    if not embeddings:
        return await fetch_and_compute_external_results(
            asset_id, asset_type, topk, embedding_store, collection_name
        )

    for vector in embeddings:
        search_results = embedding_store.retrieve_topk_document_ids(
            model=None,
            query_text=None,
            collection_name=collection_name,
            topk=topk,
            filter="",
            precomputed_embedding=vector,
        )
        if search_results:
            similar_query = SimilarQuery(asset_id=asset_id, result_set=search_results)
            return similar_query.map_to_response()

    raise HTTPException(status_code=404, detail="Recommender query not found.")


async def fetch_and_compute_external_results(
    asset_id: str,
    asset_type: AssetType,
    topk: int,
    embedding_store,
    collection_name: str,
) -> dict:

    logging.info(
        f"No embeddings found for asset_id '{asset_id}'. Fetching external data..."
    )
    url = f"{settings.AIOD.URL}{asset_type.value}/v1/{asset_id}"
    response = requests.get(url, params={"schema": "aiod"})
    if response.status_code != 200:
        raise HTTPException(
            status_code=404,
            detail="No embeddings found and external API request failed.",
        )

    dataset_info = response.json()
    processed_data = ConvertJsonToString.stringify(dataset_info)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and settings.USE_GPU else "cpu"
    )
    model = AiModel(device=device)
    tensor = model.compute_asset_embeddings([processed_data])
    embeddings_list = [emb.cpu().numpy().tolist() for emb in tensor]

    merged_results = merge_search_results_from_embeddings(
        embedding_store, collection_name, topk, embeddings_list
    )
    similar_query = SimilarQuery(asset_id=asset_id, result_set=merged_results)
    return similar_query.map_to_response()


def merge_search_results_from_embeddings(
    embedding_store, collection_name: str, topk: int, embeddings_list: list
) -> SearchResults:

    merged_doc_ids = []
    merged_distances = []

    for (
        emb
    ) in embeddings_list:  # can return multiple lists (if there are multiple chunks)
        for doc in emb:
            search_results = embedding_store.retrieve_topk_document_ids(
                model=None,
                query_text=None,
                collection_name=collection_name,
                topk=topk,
                filter="",
                precomputed_embedding=doc,
            )

            if search_results:
                merged_doc_ids.extend(search_results.doc_ids)
                merged_distances.extend(search_results.distances)

    if not merged_doc_ids:
        raise HTTPException(
            status_code=404, detail="Recommender query not found after external fetch."
        )

    all_results = list(zip(merged_doc_ids, merged_distances))
    sorted_results = sorted(all_results, key=lambda x: x[1], reverse=True)[:topk]
    final_doc_ids, final_distances = (
        zip(*sorted_results) if sorted_results else ([], [])
    )

    return SearchResults(doc_ids=list(final_doc_ids), distances=list(final_distances))
