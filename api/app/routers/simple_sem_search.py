from typing import Annotated
from uuid import UUID

from app.models.query import SimpleUserQuery
from app.routers.sem_search import (
    get_query_results,
    submit_query,
    validate_query_endpoint_arguments_or_raise,
)
from app.schemas.enums import AssetType
from app.schemas.query import SimpleUserQueryResponse
from app.services.database import Database
from fastapi import APIRouter, Depends, Path, Query
from fastapi.responses import RedirectResponse

router = APIRouter(prefix="/query", tags=["query"])


@router.post("")
async def submit_simple_query(
    database: Annotated[Database, Depends(Database)],
    query: str = Query(..., max_length=100, min_length=1, description="User query"),
    asset_type: AssetType = Query(..., description="Asset type"),
    topk: int = Query(
        default=10, gt=0, le=100, description="Number of results to search for"
    ),
) -> RedirectResponse:
    validate_query_endpoint_arguments_or_raise(
        query, asset_type, database, apply_filtering=False
    )
    user_query = SimpleUserQuery(
        orig_query=query.strip(), asset_type=asset_type, topk=topk
    )
    return await submit_query(user_query, database, router)


@router.get("/{query_id}/result")
async def get_simple_query_result(
    database: Annotated[Database, Depends(Database)],
    query_id: UUID = Path(..., description="Valid query ID"),
) -> SimpleUserQueryResponse:
    return await get_query_results(query_id, database, SimpleUserQuery)
