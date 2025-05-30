from typing import Type, TypeVar
from uuid import UUID

from fastapi import HTTPException

from app.config import settings
from app.models.query import BaseUserQuery
from app.schemas.enums import BaseAssetType
from app.schemas.query import BaseUserQueryResponse
from app.services.database import Database
from app.services.threads.search_thread import QUERY_QUEUE

Response = TypeVar("Response", bound=BaseUserQueryResponse)


async def submit_query(user_query: BaseUserQuery, database: Database) -> str:
    database.insert(user_query)
    QUERY_QUEUE.put((user_query.id, type(user_query)))
    return user_query.id


async def get_query_results(
    query_id: UUID, database: Database, query_type: Type[BaseUserQuery]
) -> Response:
    user_query = database.find_by_id(query_type, id=str(query_id))
    if user_query is None:
        raise HTTPException(
            status_code=404, detail="Requested query doesn't exist or has been deleted."
        )
    if user_query.is_expired:
        raise HTTPException(status_code=410, detail="Requested query has expired.")
    return user_query.map_to_response()


def validate_query_or_raise(query: str) -> None:
    if not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Invalid/empty user query",
        )


def validate_asset_type_or_raise(
    asset_type: BaseAssetType, database: Database, apply_filtering: bool
) -> None:
    valid_asset_types = (
        settings.AIOD.ASSET_TYPES_FOR_METADATA_EXTRACTION
        if apply_filtering
        else settings.AIOD.ASSET_TYPES
    )
    if apply_filtering and asset_type.is_all():
        raise HTTPException(
            status_code=400,
            detail=f"You need to specify a specific asset type to perform a filtered search",
        )

    if not asset_type.is_all():
        supp_asset_type = asset_type.to_SupportedAssetType()

        if supp_asset_type not in valid_asset_types:
            raise HTTPException(
                status_code=404,
                detail=f"We currently do not support asset type '{supp_asset_type.value}'",
            )
        if database.get_first_asset_collection_by_type(supp_asset_type) is None:
            raise HTTPException(
                status_code=501,
                detail=f"The database for the asset type '{supp_asset_type.value}' has yet to be built. Try again later...",
            )
