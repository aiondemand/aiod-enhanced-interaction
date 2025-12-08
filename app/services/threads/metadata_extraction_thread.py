import logging
import os
import threading

from beanie.odm.queries.find import FindMany
from beanie.operators import In
from app.config import settings
from app.models.asset_for_metadata_extraction import AssetForMetadataExtraction
from app.schemas.enums import SupportedAssetType
from app.services.embedding_store import EmbeddingStore, MilvusEmbeddingStore
from app.services.metadata_filtering.metadata_extraction_agent import MetadataExtractionWrapper
from app.services.resilience import LocalServiceUnavailableException

job_lock = threading.Lock()


BATCH_SIZE = 100


# Job for extracting metadata from AIoD assets using an LLM and updating corresponding documents within Milvus database
async def extract_metadata_for_assets_wrapper() -> None:
    if job_lock.acquire(blocking=False):
        try:
            logging.info(
                "[RECURRING METADATA EXTRACTION] Scheduled task for extracting asset metadata has started"
            )
            await extract_metadata_for_assets()
            logging.info("Scheduled task for extracting asset metadata has ended.")
        finally:
            job_lock.release()
    else:
        logging.info(
            "Scheduled task for metadata extraction skipped (previous task is still running)"
        )


async def extract_metadata_for_assets() -> None:
    embedding_store = MilvusEmbeddingStore()

    for asset_type in settings.AIOD.ASSET_TYPES_FOR_METADATA_EXTRACTION:
        logging.info(f"\tExtracting metadata for asset type: {asset_type.value}")
        try:
            await process_assets_for_metadata_extraction(
                embedding_store=embedding_store,
                asset_type=asset_type,
            )
        except LocalServiceUnavailableException as e:
            logging.error(e)
            logging.error(
                "The above error has been encountered in the metadata extraction thread. "
                + "Entire Application is being terminated now"
            )
            os._exit(1)
        except Exception as e:
            # Non-critical errors - log and continue with other asset types
            logging.error(e)
            logging.error(
                f"The above error has been encountered while processing {asset_type.value} "
                + "in the metadata extraction thread. Continuing with other asset types."
            )


async def process_assets_for_metadata_extraction(
    embedding_store: MilvusEmbeddingStore,
    asset_type: SupportedAssetType,
) -> None:
    total_updated = 0

    while True:
        assets_batch = await _prepare_mongo_asset_batch(asset_type).to_list()
        if len(assets_batch) == 0:
            break

        # Extract metadata for all assets in the batch
        assets_metadata = [
            await MetadataExtractionWrapper.extract_metadata(asset_doc.asset, asset_doc.asset_type)
            for asset_doc in assets_batch
        ]

        try:
            num_updated = await update_milvus_batch(
                embedding_store=embedding_store,
                assets_batch=assets_batch,
                assets_metadata=assets_metadata,
                asset_type=asset_type,
            )
            total_updated += num_updated
        except Exception as e:
            logging.error(f"\t\tFailed to process batch for {asset_type.value}: {e}")

        await AssetForMetadataExtraction.delete_docs(
            In(AssetForMetadataExtraction.asset_id, [asset.asset_id for asset in assets_batch])
        )

    logging.info(
        f"\tMetadata extraction for {asset_type.value} completed: {total_updated} assets updated in Milvus."
    )


async def update_milvus_batch(
    embedding_store: EmbeddingStore,
    assets_batch: list[AssetForMetadataExtraction],
    assets_metadata: list[dict],
    asset_type: SupportedAssetType,
) -> int:
    asset_ids = [asset.asset_id for asset in assets_batch]
    mongo_versions = {asset.asset_id: asset.asset_version for asset in assets_batch}
    metadata_map = {asset.asset_id: meta for asset, meta in zip(assets_batch, assets_metadata)}

    all_milvus_records = embedding_store.read_records(
        asset_type=asset_type, filter=f"asset_id in {asset_ids}", output_fields=["*"]
    )

    # Check asset versions are correct for each record
    records_to_update: list[dict] = []
    for record in all_milvus_records:
        asset_id = record["asset_id"]
        if record["asset_version"] == mongo_versions[asset_id]:
            records_to_update.append({**record, **metadata_map[asset_id]})

    if records_to_update:
        num_updated = embedding_store.upsert_records(records_to_update, asset_type)
        return num_updated
    else:
        return 0


def _prepare_mongo_asset_batch(
    asset_type: SupportedAssetType,
) -> FindMany[AssetForMetadataExtraction]:
    return (
        AssetForMetadataExtraction.find(AssetForMetadataExtraction.asset_type == asset_type)
        .sort("created_at", 1)
        .limit(BATCH_SIZE)
    )
