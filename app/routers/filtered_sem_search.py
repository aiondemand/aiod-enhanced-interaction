from typing import Annotated, Any, Type
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.models.filter import Filter
from app.models.query import FilteredUserQuery
from app.routers.sem_search import (
    get_query_results,
    submit_query,
    validate_asset_type_or_raise,
    validate_query_or_raise,
)
from app.schemas.asset_metadata.operations import SchemaOperations
from app.schemas.enums import SupportedAssetType
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
    asset_type: SupportedAssetType = Query(
        SupportedAssetType.DATASETS,
        description="Asset type eligible for metadata filtering. Currently only 'datasets' asset type works.",
    ),
    filters: Annotated[list[Filter], Field(..., max_length=5)] | None = Body(
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
    return_entire_assets: bool = Query(
        default=False,
        description="Whether to return the entire AIoD assets or only their corresponding IDs",
    ),
) -> FilteredUserQueryResponse:
    return await get_query_results(
        query_id, database, FilteredUserQuery, return_entire_assets=return_entire_assets
    )


@router.get("/schemas/get_fields")
async def get_fields_to_filter_by(
    asset_type: SupportedAssetType = Query(
        SupportedAssetType.DATASETS,
        description="Asset type we wish to create a filter for. Currently only 'datasets' asset type works.",
    ),
) -> dict:
    if asset_type not in settings.AIOD.ASSET_TYPES_FOR_METADATA_EXTRACTION:
        raise HTTPException(
            status_code=404,
            detail=f"We currently do not support asset type '{asset_type.value}'",
        )

    schema = SchemaOperations.get_asset_schema(asset_type)
    field_names = SchemaOperations.get_schema_field_names(schema)

    inner_class_dict: dict[str, Any] = {
        "__annotations__": {},
    }
    for field in field_names:
        inner_class_dict[field] = SchemaOperations.get_inner_field_info(schema, field)
        inner_class_dict["__annotations__"][field] = SchemaOperations.get_inner_annotation(
            schema, field
        )

    fields_class: Type[BaseModel] = type(
        f"Fields_{asset_type.value}", (BaseModel,), inner_class_dict
    )
    output_schema = fields_class.model_json_schema()["properties"]

    for field in output_schema.keys():
        output_schema[field].pop("default", None)
        output_schema[field].pop("title", None)

    return output_schema


@router.get("/schemas/get_filter_schema")
async def get_filter_schema(
    asset_type: SupportedAssetType = Query(
        SupportedAssetType.DATASETS,
        description="Asset type we wish to create a filter for. Currently only 'datasets' asset type works.",
    ),
    field_name: str = Query(..., description="Name of the field we wish to filter assets by"),
) -> dict:
    if asset_type not in settings.AIOD.ASSET_TYPES_FOR_METADATA_EXTRACTION:
        raise HTTPException(
            status_code=404,
            detail=f"We currently do not support asset type '{asset_type.value}'",
        )

    schema = SchemaOperations.get_asset_schema(asset_type)
    if field_name not in SchemaOperations.get_schema_field_names(schema):
        raise HTTPException(
            status_code=400,
            detail=f"Asset type '{asset_type.value}' does not support field name '{field_name}'",
        )

    filter_class = Filter.create_field_specific_filter_type(asset_type, field_name)
    return filter_class.model_json_schema()


async def _sumbit_filtered_query(
    database: Database,
    search_query: str,
    asset_type: SupportedAssetType,
    filters: list[Filter] | None,
    topk: int,
) -> str:
    validate_query_or_raise(search_query)
    validate_asset_type_or_raise(asset_type, database, apply_filtering=True)

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
