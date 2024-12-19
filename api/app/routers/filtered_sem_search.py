from typing import Annotated
from uuid import UUID

from app.models.query import FilteredUserQuery
from app.routers.sem_search import validate_query_endpoint_arguments_or_raise
from app.schemas.enums import AssetType
from app.schemas.query import FilteredUserQueryResponse
from app.services.database import Database
from app.services.threads.search_thread import QUERY_QUEUE
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.responses import RedirectResponse

router = APIRouter()


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
    # TODO later on we wish to support user-defined condtions
    # but this requires checking types and values of these conditions...
    # filters: list[Condition] | None = Body(
    #     None, description="Manually user-defined filters to apply"
    # ),
) -> RedirectResponse:
    validate_query_endpoint_arguments_or_raise(
        query, asset_type, database, apply_filtering=False
    )
    # TODO For now we set filters to None => no user-defined filters
    filters = None
    userQuery = FilteredUserQuery(
        orig_query=query.strip(),
        asset_type=asset_type,
        topk=topk,
        topic=query.strip() if filters is not None else "",
        filters=filters,
    )
    database.insert(userQuery)
    QUERY_QUEUE.put((userQuery.id, FilteredUserQuery))

    return RedirectResponse(f"/filtered_query/{userQuery.id}/result", status_code=202)


@router.get("/{query_id}/result")
async def get_filtered_query_result(
    database: Annotated[Database, Depends(Database)],
    query_id: UUID = Path(..., description="Valid query ID"),
) -> FilteredUserQueryResponse:
    userQuery = database.find_by_id(FilteredUserQuery, id=str(query_id))
    if userQuery is None:
        raise HTTPException(status_code=404, detail="Requested query doesn't exist.")
    return userQuery.map_to_response()
