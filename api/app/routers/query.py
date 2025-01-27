import logging
from typing import Annotated, Any
from uuid import UUID

import numpy as np
from app.models.query import UserQuery
from app.schemas.enums import AssetType
from app.schemas.query import UserQueryResponse
from app.services.database import Database
from app.services.threads.search_thread import QUERY_QUEUE
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.responses import RedirectResponse
from starlette.responses import JSONResponse

from api.app.config import settings
from api.app.schemas.search_results import SearchResults
from app.services.embedding_store import Milvus_EmbeddingStore



router = APIRouter()


@router.post("")
async def submit_query(
    database: Annotated[Database, Depends(Database)],
    query: str = Query(..., max_length=100, description="User query"),
    asset_type: AssetType = Query(..., description="Asset type"),
    topk: int = Query(
        default=10, gt=0, le=100, description="Number of results to search for"
    ),
) -> RedirectResponse:
    asset_col = database.get_asset_collection_by_type(asset_type)
    if asset_col is None:
        raise HTTPException(
            status_code=501,
            detail=f"The database for the asset type '{asset_type.value}' has yet to be built",
        )

    userQuery = UserQuery(query=query, asset_type=asset_type, topk=topk)
    database.queries.insert(userQuery)
    QUERY_QUEUE.put(userQuery.id)

    return RedirectResponse(f"/query/{userQuery.id}/result", status_code=202)


@router.get("/{query_id}/result")
async def get_query_result(
    database: Annotated[Database, Depends(Database)],
    query_id: UUID = Path(..., description="Valid query ID"),
) -> UserQueryResponse:
    userQuery = database.queries.find_by_id(str(query_id))
    if userQuery is None:
        raise HTTPException(status_code=404, detail="Requested query doesn't exist.")
    return userQuery.map_to_response()


@router.get("/recommender/{asset_id}", response_class=JSONResponse)
async def get_query_similar_result(
    asset_type: AssetType = Query(..., description="Asset type"),
    asset_id: str = Path(..., description="Asset ID"),
    topk: int = Query(
        default=10, gt=0, le=100, description="Number of similar assets to retrieve"
    ),
) -> JSONResponse:
    try:
        embedding_store = await Milvus_EmbeddingStore.init()

        collection_name = settings.MILVUS.get_collection_name(asset_type)

        embeddings = embedding_store.get_embeddings(asset_id, collection_name)

        if embeddings is None or len(embeddings) == 0:
            raise ValueError(f"No embeddings found for asset_id '{asset_id}'.")


        vectors = [embedding.tolist() if isinstance(embedding, np.ndarray) else [float(x) for x in embedding] for embedding in embeddings]

        aggregated_embedding = np.mean(vectors, axis=0).tolist()

        search_results: SearchResults = embedding_store.retrieve_topk_document_ids(
            model=None,
            query_text=None,
            collection_name=collection_name,
            topk=topk,
            filter="",
            precomputed_embedding=aggregated_embedding
        )

        # Optionally, remove the query asset itself from the results
        similar_assets = []
        for doc_id, distance in zip(search_results.doc_ids, search_results.distances):
            if doc_id != asset_id:
                similar_assets.append({"doc_id": doc_id, "distance": distance})
            if len(similar_assets) >= topk:
                break

        return JSONResponse(content={
            "asset_id": asset_id,
            "similar_assets": similar_assets,
        })

    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logging.error(f"Error in recommender endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")