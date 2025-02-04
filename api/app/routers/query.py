import logging
from typing import Annotated
from uuid import UUID

from app.models.query import UserQuery, SimilarQuery
from app.schemas.enums import AssetType
from app.schemas.query import UserQueryResponse
from app.services.database import Database
from app.services.threads.search_thread import QUERY_QUEUE
from app.services.recommender import get_similar_query_response
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.responses import RedirectResponse
from api.app.schemas.query import SimilarQueryResponse

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


@router.get("/recommender/{asset_id}")
async def get_query_similar_result(
    asset_type: AssetType = Query(..., description="Asset type"),
    asset_id: str = Path(..., description="Asset ID"),
    topk: int = Query(
        default=10, gt=0, le=100, description="Number of similar assets to retrieve"
    ),
) -> SimilarQueryResponse:
    try:
        response = await get_similar_query_response(asset_id, asset_type, topk)
        return response
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logging.error(f"Error in recommender endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
