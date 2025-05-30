from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from fastapi.responses import RedirectResponse

from app.models.query import RecommenderUserQuery
from app.routers.sem_search import (
    get_query_results,
    submit_query,
    validate_asset_type_or_raise,
)
from app.schemas.enums import SupportedAssetType, AssetTypeQueryParam
from app.schemas.query import RecommenderUserQueryResponse
from app.services.database import Database

router = APIRouter()


@router.post("")
async def submit_recommender_query(
    database: Annotated[Database, Depends(Database)],
    asset_id: int = Query(..., ge=0, description="Asset ID"),
    asset_type: SupportedAssetType = Query(
        ..., description="Asset type of an asset to find recommendations to"
    ),
    output_asset_type: AssetTypeQueryParam = Query(
        ..., description="Output asset type of assets to return"
    ),
    topk: int = Query(default=10, gt=0, le=100, description="Number of similar assets to return"),
) -> RedirectResponse:
    query_id = await _submit_recommender_query(
        database, asset_id, asset_type, output_asset_type, topk
    )
    return RedirectResponse(url=f"/recommender/{query_id}/result", status_code=202)


@router.get("/{query_id}/result", response_model=RecommenderUserQueryResponse)
async def get_recommender_result(
    database: Annotated[Database, Depends(Database)],
    query_id: UUID,
) -> RecommenderUserQueryResponse:
    return await get_query_results(query_id, database, RecommenderUserQuery)


async def _submit_recommender_query(
    database: Database,
    asset_id: int,
    asset_type: SupportedAssetType,
    output_asset_type: AssetTypeQueryParam,
    topk: int,
) -> str:
    validate_asset_type_or_raise(asset_type, database, apply_filtering=False)
    validate_asset_type_or_raise(output_asset_type, database, apply_filtering=False)

    user_query = RecommenderUserQuery(
        asset_id=asset_id,
        asset_type=asset_type,
        output_asset_type=output_asset_type,
        topk=topk,
    )
    return await submit_query(user_query, database)
