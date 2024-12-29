import logging
import threading
from datetime import datetime, timezone

from app.config import settings
from app.models.query import FilteredUserQuery, SimpleUserQuery
from app.services.database import Database
from tinydb import Query

job_lock = threading.Lock()


# Get rid of expired queries from TindyDB database
async def delete_expired_queries_wrapper() -> None:
    if job_lock.acquire(blocking=False):
        try:
            logging.info(
                "[RECURRING TINYDB DELETE] Scheduled task for deleting expired queries from TinyDB has started."
            )

            database = Database()
            current_time = datetime.now(tz=timezone.utc)

            for asset_type in settings.AIOD.ASSET_TYPES:
                logging.info(f"\tDeleting embeddings of asset type: {asset_type.value}")
                delete_expired_queries(database, current_time)

            logging.info(
                "[RECURRING TINYDB DELETE] Scheduled task for deleting expired queries from TinyDB has ended."
            )
        finally:
            job_lock.release()
    else:
        logging.info(
            "Scheduled task for deleting expired queries skipped (previous task is still running)"
        )


def delete_expired_queries(database: Database, current_time: datetime) -> None:
    condition = Query().expires_at < current_time

    removed_ids = database.delete(SimpleUserQuery, condition)
    if len(removed_ids) > 0:
        logging.info(f"\tDeleted {len(removed_ids)} simple queries")

    database.delete(FilteredUserQuery, condition)
    if len(removed_ids) > 0:
        logging.info(f"\tDeleted {len(removed_ids)} filtered queries")
