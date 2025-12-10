import logging
import threading

from app_temp.config import settings
from app_temp.models.asset_for_metadata_extraction import AssetForMetadataExtraction
from app_temp.schemas.params import RequestParams
from app_temp.services.aiod import recursive_aiod_asset_fetch


job_lock = threading.Lock()


# Cleanup expired queries and empty asset collections from MongoDB database
async def crawl_assets_job() -> None:
    if job_lock.acquire(blocking=False):
        try:
            logging.info("[ASSET EXTRACTION] The job has started.")

            await crawl_assets()

            logging.info("[ASSET EXTRACTION] The job has ended.")
        finally:
            job_lock.release()
    else:
        pass


async def crawl_assets() -> None:
    for asset_type in settings.AIOD.ASSET_TYPES:
        logging.info(f"\t[ASSET EXTRACTION] Processing asset type: {asset_type.value}")

        # Retrieve existing asset_ids from MongoDB to avoid duplicates
        existing_docs = await AssetForMetadataExtraction.find_all_docs(
            AssetForMetadataExtraction.asset_type == asset_type
        )
        existing_asset_ids = set(doc.asset_id for doc in existing_docs)
        logging.info(
            f"\t\t[ASSET EXTRACTION] Found {len(existing_asset_ids)} existing assets "
            f"for {asset_type.value} in MongoDB"
        )

        url_params = RequestParams(
            offset=settings.AIOD.START_OFFSET, limit=settings.AIOD.WINDOW_SIZE
        )
        mark_recursions: list[int] = []
        total_stored = 0
        total_skipped = 0

        while True:
            assets = recursive_aiod_asset_fetch(asset_type, url_params, mark_recursions)

            # Check if we've reached the end
            if len(assets) == 0 and len(mark_recursions) == 0:
                break

            # Skip empty pages (but continue pagination)
            if len(assets) == 0:
                url_params.offset += settings.AIOD.OFFSET_INCREMENT
                continue

            # Store only new assets to MongoDB (skip duplicates)
            skipped_this_batch = 0
            for asset in assets:
                asset_id = asset["identifier"]
                if asset_id in existing_asset_ids:
                    skipped_this_batch += 1
                    total_skipped += 1
                    continue

                asset_doc = AssetForMetadataExtraction.create_asset(
                    asset, asset_type=asset_type, asset_version=0
                )
                await asset_doc.create_doc()
                existing_asset_ids.add(asset_id)
                total_stored += 1

            logging.info(
                f"\t\t[ASSET EXTRACTION] Stored {len(assets) - skipped_this_batch} new assets "
                f"(skipped {skipped_this_batch} duplicates) for {asset_type.value} "
                f"(offset={url_params.offset}, total stored={total_stored})"
            )

            # Increment offset for next page
            url_params.offset += settings.AIOD.OFFSET_INCREMENT

            # Check testing mode limit
            if settings.AIOD.TESTING and url_params.offset >= 20:
                break

        logging.info(
            f"\t[ASSET EXTRACTION] Completed asset type: {asset_type.value} "
            f"(total stored: {total_stored}, total skipped: {total_skipped})"
        )
