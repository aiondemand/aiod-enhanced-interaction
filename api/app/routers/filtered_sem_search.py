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
from fastapi import APIRouter, Body, Depends, Path, Query
from fastapi.responses import RedirectResponse

router = APIRouter(prefix="/filtered_query", tags=["filtered_query"])


@router.post("")
async def submit_filtered_query(
    database: Annotated[Database, Depends(Database)],
    query: str = Query(
        ..., max_length=100, min_length=1, description="User query with filters"
    ),
    asset_type: AssetType = Query(..., description="Asset type"),
    topk: int = Query(
        default=10, gt=0, le=100, description="Number of results to search for"
    ),
    filters: list[Filter] | None = Body(
        None, description="Manually user-defined filters to apply"
    ),
) -> RedirectResponse:
    validate_query_endpoint_arguments_or_raise(
        query, asset_type, database, apply_filtering=False
    )
    if filters is not None:
        [filter.validate_filter_or_raise(asset_type) for filter in filters]

    user_query = FilteredUserQuery(
        orig_query=query.strip(),
        asset_type=asset_type,
        topk=topk,
        topic=query.strip() if filters is not None else "",
        filters=filters,
    )
    return await submit_query(user_query, database, router)


@router.get("/{query_id}/result")
async def get_filtered_query_result(
    database: Annotated[Database, Depends(Database)],
    query_id: UUID = Path(..., description="Valid query ID"),
) -> FilteredUserQueryResponse:
    return await get_query_results(query_id, database, FilteredUserQuery)
