from typing import Annotated
from uuid import UUID

from app.models.query import SimilarUserQuery
from app.routers.sem_search import (
    get_query_results,
    submit_query,
    validate_query_endpoint_arguments_or_raise,
)
from app.schemas.enums import AssetType
from app.schemas.query import SimilarUserQueryResponse
from app.services.database import Database
from fastapi import APIRouter, Depends, Query
from fastapi.responses import RedirectResponse

router = APIRouter()


@router.post("")
async def submit_recommender_query(
    database: Annotated[Database, Depends(Database)],
    asset_id: int = Query(..., description="Asset ID"),
    asset_type: AssetType = Query(..., description="Asset type"),
    topk: int = Query(
        default=10, gt=0, le=100, description="Number of similar assets to return"
    ),
) -> RedirectResponse:
    query_id = await _submit_recommender_query(database, asset_id, asset_type, topk)
    return RedirectResponse(url=f"/recommender/{query_id}/result", status_code=202)


@router.get("/{query_id}/result", response_model=SimilarUserQueryResponse)
async def get_recommender_result(
    database: Annotated[Database, Depends(Database)],
    query_id: UUID,
) -> SimilarUserQueryResponse:
    return await get_query_results(query_id, database, SimilarUserQuery)


async def _submit_recommender_query(
    database: Database, asset_id: int, asset_type: AssetType, topk: int
) -> str:
    validate_query_endpoint_arguments_or_raise(
        query=asset_id,
        asset_type=asset_type,
        database=database,
        apply_filtering=False,
    )
    user_query = SimilarUserQuery(asset_id=asset_id, asset_type=asset_type, topk=topk)

    return await submit_query(user_query, database)
