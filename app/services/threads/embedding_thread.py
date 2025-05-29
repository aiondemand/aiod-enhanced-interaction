import gc
import logging
import os
import threading
from datetime import datetime, timezone
from functools import partial
from typing import Callable, Literal

import numpy as np
import torch
from app.config import settings
from app.models.asset_collection import AssetCollection
from app.schemas.enums import SupportedAssetType
from app.schemas.params import RequestParams
from app.services.aiod import recursive_aiod_asset_fetch
from app.services.database import Database
from app.services.embedding_store import EmbeddingStore, MilvusEmbeddingStore
from app.services.inference.model import AiModel
from app.services.inference.text_operations import (
    ConvertJsonToString,
    HuggingFaceDatasetExtractMetadata,
)
from app.services.resilience import LocalServiceUnavailableException
from torch.utils.data import DataLoader

job_lock = threading.Lock()


async def compute_embeddings_for_aiod_assets_wrapper(
    first_invocation: bool = False,
) -> None:
    if job_lock.acquire(blocking=False):
        try:
            log_msg = (
                "[INITIAL UPDATE] Initial task for computing asset embeddings has started"
                if first_invocation
                else "[RECURRING UPDATE] Scheduled task for computing asset embeddings has started"
            )
            logging.info(log_msg)

            model = AiModel(device=AiModel.get_device())
            await compute_embeddings_for_aiod_assets(model, first_invocation)
            logging.info("Scheduled task for computing asset embeddings has ended.")
        finally:
            # GPU memory cleanup
            model.to_device("cpu")
            del model
            torch.cuda.empty_cache()
            gc.collect()

            job_lock.release()
    else:
        logging.info("Scheduled task for updating skipped (previous task is still running)")


async def compute_embeddings_for_aiod_assets(model: AiModel, first_invocation: bool) -> None:
    database = Database()
    embedding_store = MilvusEmbeddingStore()

    asset_types = settings.AIOD.ASSET_TYPES
    for asset_type in asset_types:
        asset_collection = fetch_asset_collection(database, asset_type, first_invocation)
        if asset_collection is None:
            continue

        extract_metadata_func = None
        meta_extract_types = settings.AIOD.ASSET_TYPES_FOR_METADATA_EXTRACTION
        if settings.MILVUS.EXTRACT_METADATA and asset_type in meta_extract_types:
            extract_metadata_func = partial(
                HuggingFaceDatasetExtractMetadata.extract_huggingface_dataset_metadata,
                asset_type=asset_type,
            )

        logging.info(f"\tComputing embeddings for asset type: {asset_type.value}")

        try:
            process_aiod_assets_wrapper(
                model=model,
                stringify_function=partial(
                    ConvertJsonToString.extract_relevant_info, asset_type=asset_type
                ),
                extract_metadata_function=extract_metadata_func,
                embedding_store=embedding_store,
                database=database,
                asset_collection=asset_collection,
                asset_type=asset_type,
            )
        except LocalServiceUnavailableException as e:
            logging.error(e)
            logging.error(
                "The above error has been encountered in the embedding thread. "
                + "Entire Application is being terminated now"
            )
            os._exit(1)
        except Exception as e:
            # We don't wish to shutdown the application unless Milvus or Ollama is down
            # If we cannot reach AIoD, we can just skip the embedding process for that day
            logging.error(e)
            logging.error("The above error has been encountered in the embedding thread.")


def fetch_asset_collection(
    database: Database, asset_type: SupportedAssetType, first_invocation: bool
) -> AssetCollection | None:
    asset_collection = database.get_first_asset_collection_by_type(asset_type)

    if asset_collection is None:
        # DB setup
        asset_collection = AssetCollection(aiod_asset_type=asset_type)
        database.insert(asset_collection)
    elif asset_collection.last_update.finished and first_invocation is False:
        # Create a new recurring DB update
        asset_collection.add_recurring_update()
        database.upsert(asset_collection)
    elif asset_collection.last_update.finished:
        # The last DB update was successful, we skip this asset in the
        # first invocation
        return None
    else:
        # The last DB update has not been finished yet, lets continue with
        # that one...
        pass

    return asset_collection


def process_aiod_assets_wrapper(
    model: AiModel,
    stringify_function: Callable[[dict], str],
    extract_metadata_function: Callable[[dict], dict] | None,
    embedding_store: EmbeddingStore,
    database: Database,
    asset_collection: AssetCollection,
    asset_type: SupportedAssetType,
) -> None:
    existing_asset_ids_from_past = embedding_store.get_all_asset_ids(asset_type)
    newly_added_asset_ids: list[int] = []

    last_update = asset_collection.last_update
    last_db_sync_datetime: datetime | None = getattr(last_update, "from_time", None)
    query_from_time: datetime | None = last_db_sync_datetime

    last_db_sync_datetime = (
        last_db_sync_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
        if last_db_sync_datetime is not None
        else None
    )

    # if it's Nth day of the month, we wish to iterate over all the data just in case
    # we have missed some assets due to large number of assets having been deleted in the past
    all_assets_day = settings.AIOD.DAY_IN_MONTH_FOR_TRAVERSING_ALL_AIOD_ASSETS
    if last_update.to_time.day == all_assets_day and last_db_sync_datetime is not None:
        query_from_time = None
        logging.info("\t\tIterating over entire database (recurring update)")

    url_params = RequestParams(
        offset=last_update.aiod_asset_offset,
        limit=settings.AIOD.WINDOW_SIZE,
        from_time=query_from_time,
        to_time=last_update.to_time,
    )
    if url_params.offset > 0:
        logging.info(f"\t\tContinue asset embedding process from asset offset={url_params.offset}")

    while True:
        assets_to_add, asset_ids_to_remove = get_assets_to_add_and_delete(
            asset_type,
            url_params,
            existing_asset_ids_from_past=existing_asset_ids_from_past,
            newly_added_asset_ids=newly_added_asset_ids,
            last_db_sync_datetime=last_db_sync_datetime,
        )
        if assets_to_add is None or asset_ids_to_remove is None:
            break

        # Remove embeddings associated with old versions of assets
        num_emb_removed = 0
        if len(asset_ids_to_remove) > 0:
            num_emb_removed = embedding_store.remove_embeddings(asset_ids_to_remove, asset_type)
            existing_asset_ids_from_past = np.array(existing_asset_ids_from_past)[
                ~np.isin(existing_asset_ids_from_past, asset_ids_to_remove)
            ].tolist()

        # Add embeddings of new assets or of new iteration of assets
        # we have just deleted
        num_emb_added = 0
        if len(assets_to_add) > 0:
            stringified_assets = [stringify_function(obj) for obj in assets_to_add]
            asset_ids = [obj["identifier"] for obj in assets_to_add]

            metadata: list[dict] = [{} for _ in assets_to_add]
            if extract_metadata_function is not None:
                metadata = [extract_metadata_function(obj) for obj in assets_to_add]

            data = [
                (obj, id, meta) for obj, id, meta in zip(stringified_assets, asset_ids, metadata)
            ]
            loader = DataLoader(
                data,
                collate_fn=lambda batch: list(zip(*batch)),
                batch_size=settings.MODEL_BATCH_SIZE,
                num_workers=0,
            )
            num_emb_added = embedding_store.store_embeddings(
                model,
                loader,
                asset_type=asset_type,
                milvus_batch_size=settings.MILVUS.BATCH_SIZE,
            )
            newly_added_asset_ids += asset_ids

        asset_collection.update(embeddings_added=num_emb_added, embeddings_removed=num_emb_removed)
        database.upsert(asset_collection)

        # during the traversal of AIoD assets, some of them may be deleted in between
        # which would make us skip some assets if we were to use tradinational
        # pagination without any overlap, hence the need for an overlap
        url_params.offset += settings.AIOD.OFFSET_INCREMENT

    asset_collection.finish()
    database.upsert(asset_collection)


def get_assets_to_add_and_delete(
    asset_type: SupportedAssetType,
    url_params: RequestParams,
    existing_asset_ids_from_past: list[int],
    newly_added_asset_ids: list[int],
    last_db_sync_datetime: datetime | None,
) -> tuple[list[dict] | None, list[int] | None]:
    mark_recursions: list[int] = []
    assets = recursive_aiod_asset_fetch(asset_type, url_params, mark_recursions)

    if len(assets) == 0 and len(mark_recursions) == 0:
        # We have reached the end of the AIoD database
        return None, None
    if len(assets) == 0:
        # The last page contained all but valid data
        # We need to jump to a next page
        return [], []
    if settings.AIOD.TESTING and url_params.offset >= 500:
        return None, None

    asset_ids = [obj["identifier"] for obj in assets]
    modified_dates = np.array([parse_aiod_asset_date(obj, none_value="now") for obj in assets])

    # new assets to store that we have never encountered before
    new_asset_idx = np.where(
        ~np.isin(asset_ids, existing_asset_ids_from_past + newly_added_asset_ids)
    )[0]
    assets_to_add = [assets[idx] for idx in new_asset_idx]

    if last_db_sync_datetime is None:
        # This is executed during setup stage
        return assets_to_add, []

    # old assets that have been changed since the last time we embedded them
    # We skip assets that have just been computed (are found within newly_added_asset_ids),
    # otherwise we would have to recompute all the assets composing the pagination
    # overlap...
    # Old assets need to be deleted first, then they're stored in DB yet again
    updated_asset_idx = np.where(
        (np.isin(asset_ids, existing_asset_ids_from_past))
        & (modified_dates >= last_db_sync_datetime)
    )[0]
    asset_ids_to_del = [asset_ids[idx] for idx in updated_asset_idx]
    assets_to_add += [assets[idx] for idx in updated_asset_idx]

    return assets_to_add, asset_ids_to_del


def parse_aiod_asset_date(
    asset: dict,
    field: str = "date_modified",
    none_value: Literal["none", "now", "zero"] = "none",
) -> datetime | None:
    string_time = asset.get("aiod_entry", {}).get(field, None)

    if string_time is not None:
        return datetime.fromisoformat(string_time).replace(tzinfo=timezone.utc)
    else:
        if none_value == "none":
            return None
        elif none_value == "now":
            return datetime.now(tz=timezone.utc)
        elif none_value == "zero":
            return datetime.fromtimestamp(0, tz=timezone.utc)
        else:
            return None
