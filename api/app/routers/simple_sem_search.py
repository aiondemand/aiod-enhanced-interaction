from typing import Annotated
from uuid import UUID

from app.models.query import SimpleUserQuery
from app.routers.sem_search import (
    get_query_results,
    submit_query,
    validate_query_endpoint_arguments_or_raise,
)
from app.schemas.enums import AssetType
from app.schemas.query import SimpleUserQueryResponse
from app.services.database import Database
from fastapi import APIRouter, Depends, Path, Query, HTTPException
from fastapi.responses import RedirectResponse

from api.app.schemas.query import SimilarQueryResponse
from api.app.services.recommender import get_similar_query_response

router = APIRouter()


@router.post("")
async def submit_simple_query(
    database: Annotated[Database, Depends(Database)],
    search_query: str = Query(
        ..., max_length=200, min_length=1, description="User search query"
    ),
    asset_type: AssetType = Query(..., description="Asset type"),
    topk: int = Query(
        default=10, gt=0, le=100, description="Number of assets to return"
    ),
) -> RedirectResponse:
    query_id = await _sumbit_simple_query(database, search_query, asset_type, topk=topk)
    return RedirectResponse(f"/query/{query_id}/result", status_code=202)


@router.get("/{query_id}/result")
async def get_simple_query_result(
    database: Annotated[Database, Depends(Database)],
    query_id: UUID = Path(..., description="Valid query ID"),
) -> SimpleUserQueryResponse:
    return await get_query_results(query_id, database, SimpleUserQuery)


async def _sumbit_simple_query(
    database: Database, search_query: str, asset_type: AssetType, topk: int
) -> str:
    validate_query_endpoint_arguments_or_raise(
        search_query,
        asset_type,
        database,
        apply_filtering=False,
    )
    user_query = SimpleUserQuery(
        search_query=search_query.strip(), asset_type=asset_type, topk=topk
    )

    return await submit_query(user_query, database)



@router.get("/recommender/{asset_id}")
async def get_similar_query_result(
    asset_type: AssetType = Query(..., description="Asset type"),
    asset_id: str = Path(..., description="Asset ID"),
    topk: int = Query(
        default=10, gt=0, le=100, description="Number of similar assets to return"
    ),
) -> SimilarQueryResponse:
    try:
        response = await get_similar_query_response(asset_id, asset_type, topk)
        return response
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
