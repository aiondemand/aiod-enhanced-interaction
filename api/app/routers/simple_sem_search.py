from typing import Annotated
from uuid import UUID

from app.helper import check_asset_collection_validity_or_raise
from app.models.query import UserQuery
from app.schemas.enums import AssetType
from app.schemas.query import SimpleUserQueryResponse
from app.services.database import Database
from app.services.threads.search_thread import QUERY_QUEUE
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.responses import RedirectResponse

router = APIRouter()


@router.post("")
async def submit_query(
    database: Annotated[Database, Depends(Database)],
    query: str = Query(..., max_length=100, min_length=1, description="User query"),
    asset_type: AssetType = Query(..., description="Asset type"),
    topk: int = Query(
        default=10, gt=0, le=100, description="Number of results to search for"
    ),
) -> RedirectResponse:
    check_asset_collection_validity_or_raise(
        database, asset_type, apply_filtering=False
    )
    query = query.strip()
    if len(query) == 0:
        raise HTTPException(
            status_code=400,
            detail="Invalid/empty user query",
        )

    userQuery = UserQuery(
        orig_query=query, asset_type=asset_type, topk=topk, apply_filtering=False
    )
    database.queries.insert(userQuery)
    QUERY_QUEUE.put(userQuery.id)

    return RedirectResponse(f"/query/{userQuery.id}/result", status_code=202)


@router.get("/{query_id}/result")
async def get_query_result(
    database: Annotated[Database, Depends(Database)],
    query_id: UUID = Path(..., description="Valid query ID"),
) -> SimpleUserQueryResponse:
    userQuery = database.queries.find_by_id(str(query_id))
    if userQuery is None:
        raise HTTPException(status_code=404, detail="Requested query doesn't exist.")
    return userQuery.map_to_response()
