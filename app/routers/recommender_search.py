from beanie import PydanticObjectId
from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import RedirectResponse

from app.config import settings
from app.models.asset_collection import AssetCollection
from app.models.query import RecommenderUserQuery
from app.routers.sem_search import (
    get_query_results,
    submit_query,
    validate_query_endpoint_arguments_or_raise,
)
from app.schemas.enums import AssetType
from app.schemas.query import RecommenderUserQueryResponse

router = APIRouter()


@router.post("")
async def submit_recommender_query(
    asset_id: int = Query(..., description="Asset ID"),
    asset_type: AssetType = Query(..., description="Asset type"),
    output_asset_type: AssetType = Query(..., description="Output Asset type"),
    topk: int = Query(default=10, gt=0, le=100, description="Number of similar assets to return"),
) -> RedirectResponse:
    query_id = await _submit_recommender_query(asset_id, asset_type, output_asset_type, topk)
    return RedirectResponse(url=f"/recommender/{query_id}/result", status_code=202)


@router.get("/{query_id}/result", response_model=RecommenderUserQueryResponse)
async def get_recommender_result(
    query_id: PydanticObjectId = Path(..., description="Valid query ID"),
) -> RecommenderUserQueryResponse:
    return await get_query_results(query_id, RecommenderUserQuery)


async def _submit_recommender_query(
    asset_id: int,
    asset_type: AssetType,
    output_asset_type: AssetType,
    topk: int,
) -> str:
    await validate_query_endpoint_arguments_or_raise(
        query=asset_id,
        asset_type=asset_type,
        apply_filtering=False,
    )

    # TODO this should be revised too
    # validation output_asset_type
    if output_asset_type not in settings.AIOD.ASSET_TYPES:
        raise HTTPException(
            status_code=404,
            detail=f"We currently do not support asset type '{output_asset_type.value}'",
        )

    asset_col = await AssetCollection.get_first_object_by_asset_type(output_asset_type)
    if asset_col is None:
        raise HTTPException(
            status_code=501,
            detail=f"The database for the asset type '{output_asset_type.value}' has yet to be built. Try again later...",
        )

    user_query = RecommenderUserQuery(
        asset_id=asset_id,
        asset_type=asset_type,
        output_asset_type=output_asset_type,
        topk=topk,
    )

    return await submit_query(user_query)
