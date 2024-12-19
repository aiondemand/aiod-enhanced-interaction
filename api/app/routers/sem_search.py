from app.config import settings
from app.schemas.enums import AssetType
from app.services.database import Database
from fastapi import HTTPException

# TODO later extract common code from query endpoints here

# async def submit_query(
#     user_query: BaseUserQuery,
#     database: Database
# ) -> RedirectResponse:
#     userQuery = SimpleUserQuery(
#         orig_query=query.strip(), asset_type=asset_type, topk=topk
#     )
#     database.insert(userQuery)
#     QUERY_QUEUE.put((userQuery.id, SimpleUserQuery))

#     # TODO make this router dependent
#     return RedirectResponse(f"/query/{userQuery.id}/result", status_code=202)
#     pass


async def get_query_results():
    pass


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
