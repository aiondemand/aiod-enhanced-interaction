from typing import Type, TypeVar

from beanie import PydanticObjectId
from fastapi import HTTPException

from app.config import settings
from app.models.asset_collection import AssetCollection
from app.models.query import BaseUserQuery
from app.schemas.enums import AssetType
from app.schemas.query import BaseUserQueryResponse
from app.services.threads.search_thread import QUERY_QUEUE

Response = TypeVar("Response", bound=BaseUserQueryResponse)


async def submit_query(user_query: BaseUserQuery) -> str:
    await user_query.create()
    QUERY_QUEUE.put((user_query.id, type(user_query)))
    return user_query.id


async def get_query_results(
    query_id: PydanticObjectId, query_type: Type[BaseUserQuery]
) -> Response:
    user_query = await query_type.get(query_id)
    if user_query is None:
        raise HTTPException(
            status_code=404, detail="Requested query doesn't exist or has been deleted."
        )
    if user_query.is_expired:
        raise HTTPException(status_code=410, detail="Requested query has expired.")
    return user_query.map_to_response()


async def validate_query_endpoint_arguments_or_raise(
    query: str | int, asset_type: AssetType, apply_filtering: bool
) -> None:
    valid_asset_types = (
        settings.AIOD.ASSET_TYPES_FOR_METADATA_EXTRACTION
        if apply_filtering
        else settings.AIOD.ASSET_TYPES
    )
    if asset_type not in valid_asset_types:
        raise HTTPException(
            status_code=404,
            detail=f"We currently do not support asset type '{asset_type.value}'",
        )
    asset_col = await AssetCollection.get_first_object_by_asset_type(asset_type)
    if asset_col is None:
        raise HTTPException(
            status_code=501,
            detail=f"The database for the asset type '{asset_type.value}' has yet to be built. Try again later...",
        )
    # TODO
    # It needs to be revised.
    if isinstance(query, str):
        if not query.strip():
            raise HTTPException(
                status_code=400,
                detail="Invalid/empty user query",
            )
    elif isinstance(query, int):
        if query is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid asset id. It must be a positive integer.",
            )
    else:
        raise HTTPException(
            status_code=400,
            detail="Query must be a non-empty string or a positive integer asset id.",
        )
