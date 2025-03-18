from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query
from fastapi.responses import RedirectResponse
from pydantic import conlist

from app.config import settings
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

router = APIRouter()


def get_body_examples_argument() -> dict:
    return {
        "extract_with_llm": {
            "summary": "No manually defined filters by a user",
            "description": "If the body is empty, an LLM is used to extract the filters",
            "value": [],
        },
        "manually_defined": {
            "summary": "Manually defined filters by a user",
            "description": "If the body contains filters, an LLM is not used for parsing query. Instead these manually user-defined filters are used.",
            "value": Filter.get_body_examples(),
        },
    }


@router.post("")
async def submit_filtered_query(
    database: Annotated[Database, Depends(Database)],
    search_query: str = Query(
        ..., max_length=200, min_length=1, description="User search query with filters"
    ),
    asset_type: AssetType = Query(
        AssetType.DATASETS,
        description="Asset type eligible for metadata filtering. Currently only 'datasets' asset type works.",
    ),
    filters: conlist(Filter, max_length=5) | None = Body(
        None,
        description="Manually user-defined filters to apply",
        openapi_examples=get_body_examples_argument(),
    ),
    topk: int = Query(default=10, gt=0, le=100, description="Number of assets to return"),
) -> RedirectResponse:
    query_id = await _sumbit_filtered_query(database, search_query, asset_type, filters, topk=topk)
    return RedirectResponse(f"/filtered_query/{query_id}/result", status_code=202)


@router.get("/{query_id}/result")
async def get_filtered_query_result(
    database: Annotated[Database, Depends(Database)],
    query_id: UUID = Path(..., description="Valid query ID"),
) -> FilteredUserQueryResponse:
    return await get_query_results(query_id, database, FilteredUserQuery)


@router.get("/schemas/get_fields")
async def get_fields_to_filter_by(
    asset_type: AssetType = Query(
        AssetType.DATASETS,
        description="Asset type we wish to create a filter for. Currently only 'datasets' asset type works.",
    ),
) -> dict:
    if asset_type not in settings.AIOD.ASSET_TYPES_FOR_METADATA_EXTRACTION:
        raise HTTPException(
            status_code=404,
            detail=f"We currently do not support asset type '{asset_type.value}'",
        )

    # TODO return a list of fields that can be filtered by for a given asset type
    # I suppose we can add their their respective types as well...

    # TODO problem -> How am I supposed to depict value restrictions for a field, if they're guarded by field_validator decorators only?

    # TODO start off with a simple list of fields without their respective types

    raise NotImplementedError


@router.get("/schemas/get_filter_schema")
async def get_filter_schema(
    asset_type: AssetType = Query(
        AssetType.DATASETS,
        description="Asset type we wish to create a filter for. Currently only 'datasets' asset type works.",
    ),
    field_name: str = Query(..., description="Name of the field we wish to filter assets by"),
) -> dict:
    if asset_type not in settings.AIOD.ASSET_TYPES_FOR_METADATA_EXTRACTION:
        raise HTTPException(
            status_code=404,
            detail=f"We currently do not support asset type '{asset_type.value}'",
        )

    # TODO initial checks
    # 1. check if asset_type is supported
    # 2. check if field_name is supported

    # TODO create Pydantic class for an asset-specific filter, and then do .model_dump() to get the schema spec

    raise NotImplementedError


async def _sumbit_filtered_query(
    database: Database,
    search_query: str,
    asset_type: AssetType,
    filters: list[Filter] | None,
    topk: int,
) -> str:
    validate_query_endpoint_arguments_or_raise(
        search_query, asset_type, database, apply_filtering=True
    )
    if filters:
        for filter in filters:
            filter.validate_filter_or_raise(asset_type)

    user_query = FilteredUserQuery(
        search_query=search_query.strip(),
        asset_type=asset_type,
        topk=topk,
        filters=filters if filters else None,
    )
    return await submit_query(user_query, database)


# TODO endpoints defining schemas for metadata filtering are required
# so that a user has an idea how he can build filters manually...
