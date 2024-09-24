from typing import Annotated

from app.models.query import UserQuery
from app.schemas.query import UserQueryResponse
from app.services.database import UserQueryDatabase
from app.services.threads.search_thread import QUERY_QUEUE
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse

router = APIRouter()


@router.post("/")
async def submit_query(
    query: str, query_database: Annotated[UserQueryDatabase, Depends(UserQueryDatabase)]
) -> RedirectResponse:
    userQuery = UserQuery(query=query)
    query_database.insert(userQuery)
    QUERY_QUEUE.put(userQuery.id)

    return RedirectResponse(f"/query/{userQuery.id}/result/", status_code=202)


@router.get("/{query_id}/result/")
async def get_query_result(
    query_id: str,
    query_database: Annotated[UserQueryDatabase, Depends(UserQueryDatabase)],
) -> UserQueryResponse:
    userQuery = query_database.find_by_id(query_id=query_id)
    if userQuery is None:
        raise HTTPException(status_code=404, detail="Requested query doesn't exist.")
    return userQuery.map_to_response()
