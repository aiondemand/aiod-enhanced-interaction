import logging
import threading
from datetime import datetime, timezone
from typing import Type

from app.config import settings
from app.models.query import BaseUserQuery, FilteredUserQuery, RecommenderUserQuery, SimpleUserQuery
from app.services.database import Database
from tinydb import Query

job_lock = threading.Lock()


# Cleanup expired queries and empty asset collections from TinyDB database
async def tinydb_cleanup() -> None:
    if job_lock.acquire(blocking=False):
        try:
            logging.info(
                "[RECURRING TINYDB DELETE] Scheduled task for cleaning up TinyDB has started."
            )

            database = Database()
            current_time = datetime.now(tz=timezone.utc)

            # delete expired queries
            logging.info(f"\tDeleting expired queries")
            delete_expired_queries(database, current_time)

            # delete empty asset collections
            logging.info(f"\tDeleting empty asset collections")
            delete_empty_asset_collections(database)

            logging.info(
                "[RECURRING TINYDB DELETE] Scheduled task for cleaning up TinyDB has ended."
            )
        finally:
            job_lock.release()
    else:
        logging.info(
            "Scheduled task for cleaning up TinyDB skipped (previous task is still running)"
        )


def delete_expired_queries(database: Database, current_time: datetime) -> None:
    condition = Query().expires_at < current_time
    query_types_to_delete: list[Type[BaseUserQuery]] = [
        SimpleUserQuery,
        FilteredUserQuery,
        RecommenderUserQuery,
    ]

    for query_type in query_types_to_delete:
        removed_ids = database.delete(query_type, condition)
        if len(removed_ids) > 0:
            logging.info(f"\tDeleted {len(removed_ids)} {query_type.__name__} queries")


def delete_empty_asset_collections(database: Database) -> None:
    for asset_type in settings.AIOD.ASSET_TYPES:
        asset_collection = database.get_first_asset_collection_by_type(asset_type)
        if asset_collection is None:
            continue

        # removing empty recurring updates except the last one
        if len(asset_collection.recurring_updates) > 0:
            asset_collection.recurring_updates = [
                update
                for update in asset_collection.recurring_updates[:-1]
                if update.embeddings_added > 0 or update.embeddings_removed > 0
            ] + [asset_collection.recurring_updates[-1]]

            database.upsert(asset_collection)
