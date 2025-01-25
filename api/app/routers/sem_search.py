from typing import Type
from uuid import UUID

from app.config import settings
from app.models.query import BaseUserQuery
from app.schemas.enums import AssetType
from app.schemas.query import BaseUserQueryResponse
from app.services.database import Database
from app.services.threads.search_thread import QUERY_QUEUE
from fastapi import HTTPException


async def submit_query(user_query: BaseUserQuery, database: Database) -> str:
    database.insert(user_query)
    QUERY_QUEUE.put((user_query.id, type(user_query)))
    return user_query.id


async def get_query_results(
    query_id: UUID, database: Database, query_type: Type[BaseUserQuery]
) -> BaseUserQueryResponse:
    userQuery = database.find_by_id(query_type, id=str(query_id))
    if userQuery is None:
        raise HTTPException(
            status_code=404, detail="Requested query doesn't exist or has been deleted."
        )
    return userQuery.map_to_response()


def validate_query_endpoint_arguments_or_raise(
    query: str, asset_type: AssetType, database: Database, apply_filtering: bool
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
    asset_col = database.get_first_asset_collection_by_type(asset_type)
    if asset_col is None:
        raise HTTPException(
            status_code=501,
            detail=f"The database for the asset type '{asset_type.value}' has yet to be built. Try again later...",
        )
    if len(query.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="Invalid/empty user query",
        )
