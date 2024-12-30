from asyncio import Condition
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
from app.services.threads.search_thread import QUERY_CONDITIONS
from fastapi import APIRouter, Depends, Path, Query
from fastapi.responses import RedirectResponse

router = APIRouter(prefix="/query", tags=["query"])


@router.post("/blocking")
async def submit_simple_query_blocking(
    database: Annotated[Database, Depends(Database)],
    search_query: str = Query(
        ..., max_length=200, min_length=1, description="User search query"
    ),
    asset_type: AssetType = Query(..., description="Asset type"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    limit: int = Query(default=10, gt=0, le=100, description="Pagination limit"),
    return_assets: bool = Query(
        default=False, description="Return entire assets instead of their IDs only"
    ),
) -> SimpleUserQueryResponse:
    query_id = await _sumbit_simple_query(
        database, search_query, asset_type, offset, limit, return_assets
    )

    # Wait till its turn to process current query
    QUERY_CONDITIONS[query_id] = Condition()
    async with QUERY_CONDITIONS[query_id]:
        await QUERY_CONDITIONS[query_id].wait()
        QUERY_CONDITIONS.pop(query_id)

    return await get_query_results(query_id, database, SimpleUserQuery)


@router.post("")
async def submit_simple_query(
    database: Annotated[Database, Depends(Database)],
    search_query: str = Query(
        ..., max_length=200, min_length=1, description="User search query"
    ),
    asset_type: AssetType = Query(..., description="Asset type"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    limit: int = Query(default=10, gt=0, le=100, description="Pagination limit"),
    return_assets: bool = Query(
        default=False, description="Return entire assets instead of their IDs only"
    ),
) -> RedirectResponse:
    query_id = await _sumbit_simple_query(
        database, search_query, asset_type, offset, limit, return_assets
    )
    return RedirectResponse(f"/query/{query_id}/result", status_code=202)


@router.get("/{query_id}/result")
async def get_simple_query_result(
    database: Annotated[Database, Depends(Database)],
    query_id: UUID = Path(..., description="Valid query ID"),
) -> SimpleUserQueryResponse:
    return await get_query_results(query_id, database, SimpleUserQuery)


async def _sumbit_simple_query(
    database: Database,
    search_query: str,
    asset_type: AssetType,
    offset: int,
    limit: int,
    return_assets: bool,
) -> str:
    validate_query_endpoint_arguments_or_raise(
        search_query,
        asset_type,
        database,
        apply_filtering=False,
    )
    user_query = SimpleUserQuery(
        search_query=search_query.strip(),
        asset_type=asset_type,
        offset=offset,
        limit=limit,
        return_assets=return_assets,
    )

    return await submit_query(user_query, database)
