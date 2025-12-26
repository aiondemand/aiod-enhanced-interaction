from uuid import UUID
from fastapi import APIRouter, Query
from fastapi.responses import RedirectResponse

from app.models import RecommenderUserQuery
from app.routers.sem_search import (
    get_query_results,
    submit_query,
    validate_asset_type_or_raise,
)
from app.schemas.asset_id import AssetId
from app.schemas.enums import SupportedAssetType, AssetTypeQueryParam
from app.schemas.query import RecommenderUserQueryResponse


router = APIRouter()


@router.post("")
async def submit_recommender_query(
    asset_id: AssetId = Query(..., description="Asset ID", pattern=r"^[a-z]{3,4}_[a-zA-Z0-9]{24}$"),
    asset_type: SupportedAssetType = Query(
        ..., description="Asset type of an asset to find recommendations to"
    ),
    output_asset_type: AssetTypeQueryParam = Query(
        ..., description="Output asset type of assets to return"
    ),
    topk: int = Query(default=10, gt=0, le=100, description="Number of similar assets to return"),
) -> RedirectResponse:
    query_id = await _submit_recommender_query(
        asset_id,
        asset_type,
        output_asset_type,
        topk,
    )
    return RedirectResponse(url=f"/recommender/{query_id}/result", status_code=202)


@router.get("/{query_id}/result", response_model=RecommenderUserQueryResponse)
async def get_recommender_result(
    query_id: UUID,
    return_entire_assets: bool = Query(
        default=False,
        description="Whether to return the entire AIoD assets or only their corresponding IDs",
    ),
) -> RecommenderUserQueryResponse:
    return await get_query_results(
        query_id,
        RecommenderUserQuery,
        return_entire_assets=return_entire_assets,
    )


async def _submit_recommender_query(
    asset_id: AssetId,
    asset_type: SupportedAssetType,
    output_asset_type: AssetTypeQueryParam,
    topk: int,
) -> UUID:
    await validate_asset_type_or_raise(asset_type, apply_filtering=False)
    await validate_asset_type_or_raise(output_asset_type, apply_filtering=False)

    user_query = RecommenderUserQuery(
        asset_id=asset_id,
        asset_type=asset_type,
        output_asset_type=output_asset_type,
        topk=topk,
    )
    return await submit_query(user_query)
