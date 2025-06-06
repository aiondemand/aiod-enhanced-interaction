import logging
import threading
from datetime import datetime
from typing import Type

from app.config import settings
from app.models.asset_collection import AssetCollection
from app.models.query import BaseUserQuery, FilteredUserQuery, RecommenderUserQuery, SimpleUserQuery
from app.services.helper import utc_now
from app.models.mongo import MongoDocument

job_lock = threading.Lock()


# Cleanup expired queries and empty asset collections from MongoDB database
async def mongo_cleanup() -> None:
    if job_lock.acquire(blocking=False):
        try:
            current_time = utc_now()
            logging.info(
                "[RECURRING MONGODB DELETE] Scheduled task for cleaning up MongoDB has started."
            )

            # delete expired queries
            await delete_expired_queries(current_time)

            # delete empty asset collections
            await delete_empty_asset_collections()

            logging.info(
                "[RECURRING MONGODB DELETE] Scheduled task for cleaning up MongoDB has ended."
            )
        finally:
            job_lock.release()
    else:
        logging.info(
            "Scheduled task for cleaning up MongoDB skipped (previous task is still running)"
        )


async def delete_expired_queries(current_time: datetime) -> None:
    query_types_to_delete: list[Type[BaseUserQuery]] = [
        SimpleUserQuery,
        FilteredUserQuery,
        RecommenderUserQuery,
    ]

    for query_type in query_types_to_delete:
        res = await MongoDocument.delete(
            query_type,
            query_type.expires_at != None,
            query_type.expires_at < current_time,  # type: ignore[operator]
        )
        if res.deleted_count > 0:
            logging.info(f"\tDeleted {res.deleted_count} {query_type.__name__} queries")


async def delete_empty_asset_collections() -> None:
    for asset_type in settings.AIOD.ASSET_TYPES:
        asset_collection = await AssetCollection.get_first_object_by_asset_type(asset_type)
        if asset_collection is None:
            continue

        # removing empty recurring updates except the last one
        if len(asset_collection.recurring_updates) > 0:
            asset_collection.recurring_updates = [
                update
                for update in asset_collection.recurring_updates[:-1]
                if update.embeddings_added > 0 or update.embeddings_removed > 0
            ] + [asset_collection.recurring_updates[-1]]

            await MongoDocument.replace(asset_collection)
