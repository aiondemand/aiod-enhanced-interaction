from asyncio import Condition
from typing import Annotated
from uuid import UUID

from app.models.filter import Filter
from app.models.query import FilteredUserQuery
from app.routers.sem_search import (
    get_query_results,
    submit_query,
    validate_query_endpoint_arguments_or_raise,
)
from app.schemas.enums import AssetType
from app.schemas.query import FilteredUserQueryResponse
from app.services.database import Database
from app.services.threads.search_thread import QUERY_CONDITIONS
from fastapi import APIRouter, Body, Depends, Path, Query
from fastapi.responses import RedirectResponse

router = APIRouter(prefix="/filtered_query", tags=["filtered_query"])


@router.post("/blocking")
async def submit_simple_query_blocking(
    database: Annotated[Database, Depends(Database)],
    search_query: str = Query(
        ..., max_length=100, min_length=1, description="User search query with filters"
    ),
    asset_type: AssetType = Query(..., description="Asset type"),
    filters: list[Filter] | None = Body(
        None, description="Manually user-defined filters to apply"
    ),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    limit: int = Query(default=10, gt=0, le=100, description="Pagination limit"),
    return_assets: bool = Query(
        default=False, description="Return entire assets instead of their IDs only"
    ),
) -> FilteredUserQueryResponse:
    query_id = await _sumbit_filtered_query(
        database, search_query, asset_type, filters, offset, limit, return_assets
    )

    # Wait till its turn to process current query
    QUERY_CONDITIONS[query_id] = Condition()
    async with QUERY_CONDITIONS[query_id]:
        await QUERY_CONDITIONS[query_id].wait()
        QUERY_CONDITIONS.pop(query_id)

    return await get_query_results(query_id, database, FilteredUserQuery)


@router.post("")
async def submit_filtered_query(
    database: Annotated[Database, Depends(Database)],
    search_query: str = Query(
        ..., max_length=100, min_length=1, description="User search query with filters"
    ),
    asset_type: AssetType = Query(..., description="Asset type"),
    filters: list[Filter] | None = Body(
        None, description="Manually user-defined filters to apply"
    ),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    limit: int = Query(default=10, gt=0, le=100, description="Pagination limit"),
    return_assets: bool = Query(
        default=False, description="Return entire assets instead of their IDs only"
    ),
) -> RedirectResponse:
    query_id = await _sumbit_filtered_query(
        database, search_query, asset_type, filters, offset, limit, return_assets
    )
    return RedirectResponse(f"/filtered_query/{query_id}/result", status_code=202)


@router.get("/{query_id}/result")
async def get_filtered_query_result(
    database: Annotated[Database, Depends(Database)],
    query_id: UUID = Path(..., description="Valid query ID"),
) -> FilteredUserQueryResponse:
    return await get_query_results(query_id, database, FilteredUserQuery)


async def _sumbit_filtered_query(
    database: Database,
    search_query: str,
    asset_type: AssetType,
    filters: list[Filter] | None,
    offset: int,
    limit: int,
    return_assets: bool,
) -> str:
    validate_query_endpoint_arguments_or_raise(
        search_query, asset_type, database, apply_filtering=False
    )
    if filters is not None:
        [filter.validate_filter_or_raise(asset_type) for filter in filters]

    user_query = FilteredUserQuery(
        search_query=search_query.strip(),
        asset_type=asset_type,
        offset=offset,
        limit=limit,
        topic=search_query.strip() if filters is not None else "",
        filters=filters,
        return_assets=return_assets,
    )
    return await submit_query(user_query, database)
