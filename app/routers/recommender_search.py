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
from app.schemas.query import OldRecommenderUserQueryResponse, RecommenderUserQueryResponse
from app.services.database import Database

router = APIRouter()


@router.post("")
async def submit_recommender_query(
    database: Annotated[Database, Depends(Database)],
    asset_id: str = Query(..., max_length=50, description="Asset ID"),
    asset_type: SupportedAssetType = Query(
        ..., description="Asset type of an asset to find recommendations to"
    ),
    output_asset_type: AssetTypeQueryParam = Query(
        ..., description="Output asset type of assets to return"
    ),
    topk: int = Query(default=10, gt=0, le=100, description="Number of similar assets to return"),
) -> RedirectResponse:
    query_id = await _submit_recommender_query(
        database,
        asset_id,
        asset_type,
        output_asset_type,
        topk,
    )
    return RedirectResponse(url=f"/recommender/{query_id}/result", status_code=202)


@router.get("/{query_id}/result", response_model=RecommenderUserQueryResponse)
async def get_recommender_result(
    database: Annotated[Database, Depends(Database)],
    query_id: UUID,
    return_entire_assets: bool = Query(
        default=False,
        description="Whether to return the entire AIoD assets or only their corresponding IDs",
    ),
) -> RecommenderUserQueryResponse:
    return await get_query_results(
        query_id,
        database,
        RecommenderUserQuery,
        return_entire_assets=return_entire_assets,
        old_schema=False,
    )


####################################
############ OLD ROUTER ############
####################################

old_router = APIRouter()


# v1 endpoint that doesn't support searching across asset types
@old_router.post("")
async def old_submit_recommender_query(
    database: Annotated[Database, Depends(Database)],
    asset_id: str = Query(..., max_length=50, description="Asset ID"),
    asset_type: SupportedAssetType = Query(
        ..., description="Asset type of an asset to find recommendations to"
    ),
    output_asset_type: SupportedAssetType = Query(
        ..., description="Output asset type of assets to return"
    ),
    topk: int = Query(default=10, gt=0, le=100, description="Number of similar assets to return"),
) -> RedirectResponse:
    query_id = await _submit_recommender_query(
        database,
        asset_id,
        asset_type,
        AssetTypeQueryParam(output_asset_type.value),
        topk,
    )
    return RedirectResponse(url=f"/recommender/{query_id}/result", status_code=202)


# v1 endpoint that doesn't support returning the entire assets nor assets of different types
@old_router.get("/{query_id}/result", response_model=OldRecommenderUserQueryResponse)
async def old_get_recommender_result(
    database: Annotated[Database, Depends(Database)],
    query_id: UUID,
) -> OldRecommenderUserQueryResponse:
    return await get_query_results(
        query_id, database, RecommenderUserQuery, return_entire_assets=False, old_schema=True
    )


async def _submit_recommender_query(
    database: Database,
    asset_id: str,
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
