from typing import Annotated

from app.models.query import UserQuery
from app.schemas.enums import AssetType
from app.schemas.query import UserQueryResponse
from app.services.database import Database
from app.services.threads.search_thread import QUERY_QUEUE
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse

router = APIRouter()


@router.post("")
async def submit_query(
    query: str,
    asset_type: AssetType,
    database: Annotated[Database, Depends(Database)],
    topk: int = 10,
) -> RedirectResponse:
    asset_col = database.get_asset_collection_by_type(asset_type)
    if asset_col is None:
        raise HTTPException(
            status_code=501,
            detail=f"The database for the asset type '{asset_type.value}' has yet to be built",
        )

    userQuery = UserQuery(query=query, asset_type=asset_type, topk=topk)
    database.queries.insert(userQuery)
    QUERY_QUEUE.put(userQuery.id)

    return RedirectResponse(f"/query/{userQuery.id}/result/", status_code=202)


@router.get("/{query_id}/result")
async def get_query_result(
    query_id: str,
    database: Annotated[Database, Depends(Database)],
) -> UserQueryResponse:
    userQuery = database.queries.find_by_id(query_id)
    if userQuery is None:
        raise HTTPException(status_code=404, detail="Requested query doesn't exist.")
    return userQuery.map_to_response()
