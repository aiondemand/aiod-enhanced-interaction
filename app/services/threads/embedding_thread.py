from __future__ import annotations

from dataclasses import dataclass, field
import gc
import logging
import os
import threading
from datetime import datetime
from functools import partial
from typing import Callable, Literal

from beanie.operators import In
import numpy as np
import torch
from app.config import settings
from app.models.asset_collection import AssetCollection
from app.models.asset_for_metadata_extraction import AssetForMetadataExtraction
from app.schemas.asset_id import AssetId
from app.schemas.enums import SupportedAssetType
from app.schemas.params import RequestParams
from app.services.aiod import recursive_aiod_asset_fetch
from app.services.embedding_store import (
    EmbeddingStore,
    MilvusEmbeddingStore,
    RemoveEmbeddingsResponse,
)
from app.services.helper import utc_now
from app.services.inference.model import AiModel
from app.services.inference.text_operations import ConvertJsonToString
from app.services.resilience import LocalServiceUnavailableException
from torch.utils.data import DataLoader

job_lock = threading.Lock()


@dataclass
class AssetIdsAccum:
    existing_asset_ids_from_past: list[AssetId] = field(default_factory=list)
    newly_added_asset_ids: list[AssetId] = field(default_factory=list)

    @classmethod
    def build_for_embedding_thread(
        cls, embedding_store: EmbeddingStore, asset_type: SupportedAssetType
    ) -> AssetIdsAccum:
        return cls(
            existing_asset_ids_from_past=embedding_store.get_all_asset_ids(asset_type),
        )

    def add_new_ids(self, ids_to_add: list[AssetId]) -> None:
        self.newly_added_asset_ids.extend(ids_to_add)

    def remove_ids_from_past(self, ids_to_del: list[AssetId]) -> None:
        self.existing_asset_ids_from_past = np.array(self.existing_asset_ids_from_past)[
            ~np.isin(self.existing_asset_ids_from_past, ids_to_del)
        ].tolist()


async def compute_embeddings_for_aiod_assets_wrapper(
    first_invocation: bool = False,
) -> None:
    if job_lock.acquire(blocking=False):
        try:
            log_msg = (
                "[RECURRING AIOD UPDATE] Initial task for computing asset embeddings has started"
                if first_invocation
                else "[RECURRING AIOD UPDATE] Scheduled task for computing asset embeddings has started"
            )
            logging.info(log_msg)

            model = AiModel(device=AiModel.get_device())
            await compute_embeddings_for_aiod_assets(model, first_invocation)
            logging.info(
                "[RECURRING AIOD UPDATE] Scheduled task for computing asset embeddings has ended."
            )
        finally:
            # GPU memory cleanup
            model.to_device("cpu")
            del model
            torch.cuda.empty_cache()
            gc.collect()

            job_lock.release()
    else:
        logging.info(
            "[RECURRING AIOD UPDATE] Scheduled task for updating skipped (previous task is still running)"
        )


async def compute_embeddings_for_aiod_assets(model: AiModel, first_invocation: bool) -> None:
    embedding_store = MilvusEmbeddingStore()

    for asset_type in settings.AIOD.ASSET_TYPES:
        asset_collection = await fetch_asset_collection(asset_type, first_invocation)
        if asset_collection is None:
            continue
        logging.info(
            f"\t[RECURRING AIOD UPDATE] Computing embeddings for asset type: {asset_type.value}"
        )
        try:
            await process_aiod_assets_wrapper(
                model=model,
                stringify_asset_function=partial(
                    ConvertJsonToString.extract_relevant_info, stringify=False
                ),
                embedding_store=embedding_store,
                asset_collection=asset_collection,
                asset_type=asset_type,
            )
        except LocalServiceUnavailableException as e:
            logging.error(e)
            logging.error(
                "\t[RECURRING AIOD UPDATE] The above error has been encountered in the embedding thread. "
                + "Entire Application is being terminated now"
            )
            os._exit(1)
        except Exception as e:
            # We don't wish to shutdown the application unless Milvus or Ollama is down
            # If we cannot reach AIoD, we can just skip the embedding process for that day
            logging.error(e)
            logging.error(
                "\t[RECURRING AIOD UPDATE] The above error has been encountered in the embedding thread."
            )


async def fetch_asset_collection(
    asset_type: SupportedAssetType, first_invocation: bool
) -> AssetCollection | None:
    asset_collection = await AssetCollection.get_first_object_by_asset_type(asset_type)

    if asset_collection is None:
        # DB setup
        asset_collection = AssetCollection(aiod_asset_type=asset_type)
        await asset_collection.create_doc()
    elif asset_collection.last_update.finished and first_invocation is False:
        # Create a new recurring DB update
        asset_collection.add_recurring_update()
        await asset_collection.replace_doc()
    elif asset_collection.last_update.finished:
        # The last DB update was successful, we skip this asset in the
        # first invocation
        return None
    else:
        # The last DB update has not been finished yet, lets continue with
        # that one...
        pass

    return asset_collection


async def process_aiod_assets_wrapper(
    model: AiModel,
    stringify_asset_function: Callable[[dict, SupportedAssetType], str],
    embedding_store: EmbeddingStore,
    asset_collection: AssetCollection,
    asset_type: SupportedAssetType,
) -> None:
    asset_ids_accum = AssetIdsAccum.build_for_embedding_thread(embedding_store, asset_type)

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
        logging.info(
            "\t\t[RECURRING AIOD UPDATE] Iterating over entire database (monthly occurrence)"
        )

    url_params = RequestParams(
        offset=last_update.aiod_asset_offset,
        from_time=query_from_time,
        to_time=last_update.to_time,
    )
    if url_params.offset > 0:
        logging.info(
            f"\t\t[RECURRING AIOD UPDATE] Continue asset embedding process from asset offset={url_params.offset}"
        )

    while True:
        assets_to_add, asset_ids_to_update = get_assets_to_add_and_update(
            asset_type,
            url_params,
            asset_ids_accum=asset_ids_accum,
            last_db_sync_datetime=last_db_sync_datetime,
        )
        if assets_to_add is None or asset_ids_to_update is None:
            break

        emb_removed_response = await _remove_assets_to_update(
            asset_ids_to_update,
            asset_ids_accum=asset_ids_accum,
            embedding_store=embedding_store,
            asset_type=asset_type,
        )
        num_emb_added = await _insert_assets(
            assets_to_add,
            updated_asset_versions=emb_removed_response.asset_versions,
            asset_ids_accum=asset_ids_accum,
            model=model,
            embedding_store=embedding_store,
            asset_type=asset_type,
            stringify_asset_function=stringify_asset_function,
        )

        asset_collection.update(
            embeddings_added=num_emb_added, embeddings_removed=emb_removed_response.emb_delete_count
        )
        await asset_collection.replace_doc()

        # during the traversal of AIoD assets, some of them may be deleted in between
        # which would make us skip some assets if we were to use tradinational
        # pagination without any overlap, hence the need for an overlap
        url_params.offset += settings.AIOD.OFFSET_INCREMENT

    asset_collection.finish()
    await asset_collection.replace_doc()

    total_added = asset_collection.last_update.embeddings_added
    total_removed = getattr(asset_collection.last_update, "embeddings_removed", 0)

    logging.info(
        f"\t\t[RECURRING AIOD UPDATE] Report (asset_type={asset_type.value}): Embeddings added: {total_added} | Embeddings removed: {total_removed}"
    )


async def _remove_assets_to_update(
    asset_ids_to_update: list[AssetId],
    asset_ids_accum: AssetIdsAccum,
    embedding_store: EmbeddingStore,
    asset_type: SupportedAssetType,
) -> RemoveEmbeddingsResponse:
    # remove old embeddings (new assets may have a different number of chunks) that have been updated
    if len(asset_ids_to_update) == 0:
        return RemoveEmbeddingsResponse()

    emb_removed_response = embedding_store.remove_embeddings(asset_ids_to_update, asset_type)
    asset_ids_accum.remove_ids_from_past(asset_ids_to_update)

    # Remove assets from MongoDB if they exist (AssetForMetadataExtraction collection)
    if settings.extracts_metadata_from_asset(asset_type):
        await AssetForMetadataExtraction.delete_docs(
            In(AssetForMetadataExtraction.asset_id, asset_ids_to_update)
        )

    return emb_removed_response


async def _insert_assets(
    assets_to_add: list[dict],
    updated_asset_versions: list[int],
    asset_ids_accum: AssetIdsAccum,
    model: AiModel,
    embedding_store: EmbeddingStore,
    asset_type: SupportedAssetType,
    stringify_asset_function: Callable[[dict, SupportedAssetType], str],
) -> int:
    # Add embeddings of new assets or of the new iteration of assets we have just deleted
    if len(assets_to_add) == 0:
        return 0

    stringified_assets = [stringify_asset_function(obj, asset_type) for obj in assets_to_add]
    asset_ids = [obj["identifier"] for obj in assets_to_add]

    # Only metadata we wish to pass here is the asset_version
    asset_versions: list[dict] = (
        [
            {"asset_version": 0} for _ in range(len(assets_to_add) - len(updated_asset_versions))
        ]  # asset_version=0 => completely new asset
        + [
            {"asset_version": version + 1} for version in updated_asset_versions
        ]  # incremented old asset versions for assets to update (that have just been deleted)
    )

    data = [
        (obj, id, version)
        for obj, id, version in zip(stringified_assets, asset_ids, asset_versions)
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
    asset_ids_accum.add_new_ids(asset_ids)

    # Update MongoDB database (AssetForMetadataExtraction collection)
    if settings.extracts_metadata_from_asset(asset_type):
        [
            await AssetForMetadataExtraction.create_asset(
                asset, asset_type=asset_type, asset_version=version["asset_version"]
            ).create_doc()
            for asset, version in zip(assets_to_add, asset_versions)
        ]
    return num_emb_added


def get_assets_to_add_and_update(
    asset_type: SupportedAssetType,
    url_params: RequestParams,
    asset_ids_accum: AssetIdsAccum,
    last_db_sync_datetime: datetime | None,
) -> tuple[list[dict] | None, list[AssetId] | None]:
    mark_recursions: list[int] = []
    assets = recursive_aiod_asset_fetch(asset_type, url_params, mark_recursions)

    if len(assets) == 0 and len(mark_recursions) == 0:
        # We have reached the end of the AIoD database
        return None, None
    if len(assets) == 0:
        # The last page contained all but valid data
        # We need to jump to a next page
        return [], []
    if settings.AIOD.TESTING and url_params.offset >= 20:
        return None, None

    asset_ids: list[AssetId] = [obj["identifier"] for obj in assets]
    modified_dates = np.array([parse_aiod_asset_date(obj, none_value="now") for obj in assets])

    # new assets to store that we have never encountered before
    new_asset_idx = np.where(
        ~np.isin(
            asset_ids,
            asset_ids_accum.existing_asset_ids_from_past + asset_ids_accum.newly_added_asset_ids,
        )
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
        (np.isin(asset_ids, asset_ids_accum.existing_asset_ids_from_past))
        & (modified_dates >= last_db_sync_datetime)
    )[0]
    asset_ids_to_update = [asset_ids[idx] for idx in updated_asset_idx]
    assets_to_add += [assets[idx] for idx in updated_asset_idx]

    return assets_to_add, asset_ids_to_update


def parse_aiod_asset_date(
    asset: dict,
    field: str = "date_modified",
    none_value: Literal["none", "now", "zero"] = "none",
) -> datetime | None:
    string_time = asset.get("aiod_entry", {}).get(field, None)

    if string_time is not None:
        return datetime.fromisoformat(string_time).replace(tzinfo=None)
    else:
        if none_value == "none":
            return None
        elif none_value == "now":
            return utc_now()
        elif none_value == "zero":
            return datetime.fromtimestamp(0)
        else:
            return None
