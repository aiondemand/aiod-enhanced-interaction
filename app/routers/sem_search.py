from typing import Type, TypeVar

from uuid import UUID
from fastapi import HTTPException

from app.config import settings
from app.models.asset_collection import AssetCollection
from app.models.query import BaseUserQuery
from app.schemas.enums import BaseAssetType
from app.schemas.query import BaseUserQueryResponse
from app.celery_tasks.search.tasks import search_query_task

Response = TypeVar("Response", bound=BaseUserQueryResponse)


async def submit_query(user_query: BaseUserQuery) -> UUID:
    await user_query.create_doc()
    # Submit query to Celery task queue
    query_type_name = user_query.__class__.__name__
    search_query_task.delay(str(user_query.id), query_type_name)
    return user_query.id


async def get_query_results(
    query_id: UUID,
    query_type: Type[BaseUserQuery],
    return_entire_assets: bool = False,
) -> Response:
    user_query = await query_type.get(query_id)
    if user_query is None:
        raise HTTPException(
            status_code=404, detail="Requested query doesn't exist or has been deleted."
        )
    if user_query.is_expired:
        raise HTTPException(status_code=410, detail="Requested query has expired.")
    return user_query.map_to_response(return_entire_assets)


def validate_query_or_raise(query: str) -> None:
    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Invalid/empty user query",
        )


async def validate_asset_type_or_raise(asset_type: BaseAssetType, apply_filtering: bool) -> None:
    valid_asset_types = (
        settings.AIOD.ASSET_TYPES_FOR_METADATA_EXTRACTION
        if apply_filtering
        else settings.AIOD.ASSET_TYPES
    )
    if apply_filtering and asset_type.is_all():
        raise HTTPException(
            status_code=400,
            detail=f"You need to specify a specific asset type to perform a filtered search on",
        )

    if not asset_type.is_all():
        supp_asset_type = asset_type.to_SupportedAssetType()

        if supp_asset_type not in valid_asset_types:
            raise HTTPException(
                status_code=404,
                detail=f"We currently do not support asset type '{supp_asset_type.value}'",
            )

        asset_col = await AssetCollection.get_first_object_by_asset_type(supp_asset_type)
        if asset_col is None:
            raise HTTPException(
                status_code=501,
                detail=f"The database for the asset type '{supp_asset_type.value}' has yet to be built. Try again later...",
            )
