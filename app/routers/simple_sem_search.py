from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Path, Query
from fastapi.responses import RedirectResponse

from app.models.query import SimpleUserQuery
from app.routers.sem_search import (
    get_query_results,
    submit_query,
    validate_query_or_raise,
    validate_asset_type_or_raise,
)
from app.schemas.enums import AssetTypeQueryParam, SupportedAssetType
from app.schemas.query import SimpleUserQueryResponse, OldSimpleUserQueryResponse
from app.services.database import Database

router = APIRouter()


@router.post("")
async def submit_simple_query(
    database: Annotated[Database, Depends(Database)],
    search_query: str = Query(..., max_length=200, min_length=1, description="User search query"),
    asset_type: AssetTypeQueryParam = Query(
        default=AssetTypeQueryParam.ALL, description="Asset type of assets to return"
    ),
    topk: int = Query(default=10, gt=0, le=100, description="Number of assets to return"),
) -> RedirectResponse:
    query_id = await _sumbit_simple_query(database, search_query, asset_type, topk=topk)
    return RedirectResponse(f"/query/{query_id}/result", status_code=202)


@router.get("/{query_id}/result")
async def get_simple_query_result(
    database: Annotated[Database, Depends(Database)],
    query_id: UUID = Path(..., description="Valid query ID"),
    return_entire_assets: bool = Query(
        default=False,
        description="Whether to return the entire AIoD assets or only their corresponding IDs",
    ),
) -> SimpleUserQueryResponse:
    return await get_query_results(
        query_id,
        database,
        SimpleUserQuery,
        return_entire_assets=return_entire_assets,
        old_schema=False,
    )


####################################
############ OLD ROUTER ############
####################################

old_router = APIRouter()


@old_router.post("")
async def old_submit_simple_query(
    database: Annotated[Database, Depends(Database)],
    search_query: str = Query(..., max_length=200, min_length=1, description="User search query"),
    asset_type: SupportedAssetType = Query(
        default=SupportedAssetType.DATASETS, description="Asset type of assets to return"
    ),
    topk: int = Query(default=10, gt=0, le=100, description="Number of assets to return"),
) -> RedirectResponse:
    query_id = await _sumbit_simple_query(
        database, search_query, asset_type=AssetTypeQueryParam(asset_type.value), topk=topk
    )
    return RedirectResponse(f"/query/{query_id}/result", status_code=202)


@old_router.get("/{query_id}/result")
async def old_get_simple_query_result(
    database: Annotated[Database, Depends(Database)],
    query_id: UUID = Path(..., description="Valid query ID"),
) -> OldSimpleUserQueryResponse:
    return await get_query_results(
        query_id, database, SimpleUserQuery, return_entire_assets=False, old_schema=True
    )


async def _sumbit_simple_query(
    database: Database, search_query: str, asset_type: AssetTypeQueryParam, topk: int
) -> str:
    validate_query_or_raise(search_query)
    validate_asset_type_or_raise(asset_type, database, apply_filtering=False)

    user_query = SimpleUserQuery(
        search_query=search_query.strip(),
        asset_type=asset_type,
        topk=topk,
    )
    return await submit_query(user_query, database)
