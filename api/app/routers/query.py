from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse

from app.schemas.query import QueryStatus, QueueItem
from app.schemas.SemanticSearchResults import SemanticSearchResult
from app.services.threads.search_process import QUERY_QUEUE, QueryResultsManager

router = APIRouter()


@router.post("/")
async def submit_query(
    query: str,
    query_manager: Annotated[QueryResultsManager, Depends(QueryResultsManager)],
) -> RedirectResponse:
    query_id = str(uuid4())

    query_manager.add_query(query_id)
    QUERY_QUEUE.put(QueueItem(id=query_id, query=query))
    return RedirectResponse(f"/query/{query_id}/status/", status_code=202)


@router.get("/{query_id}/status/")
async def get_query_status(
    query_id: str,
    query_manager: Annotated[QueryResultsManager, Depends(QueryResultsManager)],
) -> Any:
    query = query_manager.get_query(query_id)
    if query is None:
        raise HTTPException(status_code=404, detail="Requested query doesn't exist.")
    if query.status == QueryStatus.COMPLETED:
        return RedirectResponse(f"/query/{query_id}/result/", status_code=303)
    return query.status


@router.get("/{query_id}/result/")
async def get_query_result_endpoint(
    query_id: str,
    query_manager: Annotated[QueryResultsManager, Depends(QueryResultsManager)],
) -> SemanticSearchResult:
    query = query_manager.get_query(query_id)
    if query is None:
        raise HTTPException(status_code=404, detail="Requested query doesn't exist.")
    if query.result is None:
        raise HTTPException(
            status_code=404, detail="Results to this query are yet to be computed."
        )
    return query.result
