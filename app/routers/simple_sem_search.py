from fastapi import APIRouter, Path, Query
from fastapi.responses import RedirectResponse
from beanie import PydanticObjectId

from app.models.query import SimpleUserQuery
from app.routers.sem_search import (
    get_query_results,
    submit_query,
    validate_query_endpoint_arguments_or_raise,
)
from app.schemas.enums import AssetType
from app.schemas.query import SimpleUserQueryResponse

router = APIRouter()


@router.post("")
async def submit_simple_query(
    search_query: str = Query(..., max_length=200, min_length=1, description="User search query"),
    asset_type: AssetType = Query(..., description="Asset type"),
    topk: int = Query(default=10, gt=0, le=100, description="Number of assets to return"),
) -> RedirectResponse:
    query_id = await _sumbit_simple_query(search_query, asset_type, topk=topk)
    return RedirectResponse(f"/query/{query_id}/result", status_code=202)


@router.get("/{query_id}/result")
async def get_simple_query_result(
    query_id: PydanticObjectId = Path(..., description="Valid query ID"),
) -> SimpleUserQueryResponse:
    return await get_query_results(query_id, SimpleUserQuery)


async def _sumbit_simple_query(search_query: str, asset_type: AssetType, topk: int) -> str:
    await validate_query_endpoint_arguments_or_raise(
        search_query,
        asset_type,
        apply_filtering=False,
    )
    user_query = SimpleUserQuery(
        search_query=search_query.strip(), asset_type=asset_type, topk=topk
    )

    return await submit_query(user_query)
