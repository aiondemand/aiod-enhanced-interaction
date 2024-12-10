import gc
import logging
import threading
from datetime import datetime
from functools import partial
from time import sleep
from typing import Callable

import numpy as np
import torch
from app.config import settings
from app.helper import (
    _perform_request,
    parse_asset_date,
    translate_datetime_to_aiod_params,
)
from app.models.asset_collections import AssetCollection
from app.schemas.enums import AssetType
from app.schemas.request_params import RequestParams
from app.services.database import Database
from app.services.embedding_store import EmbeddingStore, Milvus_EmbeddingStore
from app.services.inference.model import AiModel
from app.services.inference.text_operations import ConvertJsonToString
from requests.exceptions import HTTPError
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
            await compute_embeddings_for_aiod_assets(first_invocation)
            logging.info("Scheduled task for computing asset embeddings has ended.")
        finally:
            job_lock.release()
    else:
        logging.info(
            "Scheduled task for updating skipped (previous task is still running)"
        )


async def compute_embeddings_for_aiod_assets(first_invocation: bool) -> None:
    model = AiModel(device=AiModel.get_device())
    database = Database()
    embedding_store = await Milvus_EmbeddingStore.init()

    try:
        asset_types = settings.AIOD.ASSET_TYPES
        for asset_type in asset_types:
            logging.info(f"\tComputing embeddings for asset type: {asset_type.value}")

            asset_collection = database.get_asset_collection_by_type(asset_type)
            if asset_collection is None:
                # DB setup
                asset_collection = AssetCollection(aiod_asset_type=asset_type)
                database.asset_collections.insert(asset_collection)
            elif asset_collection.last_update.finished and first_invocation is False:
                # Create a new recurring DB update
                asset_collection.add_recurring_update()
                database.asset_collections.upsert(asset_collection)
            elif asset_collection.last_update.finished:
                # The last DB update was sucessful, we skip this asset in the
                # first invocation
                continue
            else:
                # The last DB update has not been finished yet, lets continue with
                # that one...
                pass

            process_aiod_assets_wrapper(
                model=model,
                stringify_function=partial(
                    ConvertJsonToString.extract_relevant_info, asset_type=asset_type
                ),
                embedding_store=embedding_store,
                database=database,
                asset_collection=asset_collection,
                asset_type=asset_type,
            )
    finally:
        model.to_device("cpu")
        del model
        torch.cuda.empty_cache()
        gc.collect()


def process_aiod_assets_wrapper(
    model: AiModel,
    stringify_function: Callable[[dict], str],
    embedding_store: EmbeddingStore,
    database: Database,
    asset_collection: AssetCollection,
    asset_type: AssetType,
) -> None:
    asset_url = settings.AIOD.get_assets_url(asset_type)
    collection_name = settings.MILVUS.get_collection_name(asset_type)
    existing_doc_ids_from_past = embedding_store.get_all_document_ids(collection_name)
    newly_added_doc_ids = []

    last_update = asset_collection.last_update
    last_db_sync_datetime: datetime = getattr(last_update, "from_time", None)
    query_from_time = last_db_sync_datetime

    last_db_sync_datetime = (
        last_db_sync_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
        if last_db_sync_datetime is not None
        else None
    )

    # if it's Nth day of the month, we wish to iterate over all the data just in case
    # we have missed some assets due to large number of assets having been deleted in the past
    all_assets_day = settings.AIOD.DAY_IN_MONTH_FOR_TRAVERSING_ALL_AIOD_ASSETS
    if last_update.to_time.day == all_assets_day:
        query_from_time = None
        logging.info("\t\tIterating over entire database (recurring update)")

    url_params = RequestParams(
        offset=last_update.aiod_asset_offset,
        limit=settings.AIOD.WINDOW_SIZE,
        from_time=query_from_time,
        to_time=last_update.to_time,
    )
    if url_params.offset > 0:
        logging.info(
            f"\t\tContinue asset embedding process from asset offset={url_params.offset}"
        )

    while True:
        assets_to_add, asset_ids_to_remove = get_assets_to_add_and_delete(
            asset_url,
            url_params,
            existing_doc_ids_from_past=existing_doc_ids_from_past,
            newly_added_doc_ids=newly_added_doc_ids,
            last_db_sync_datetime=last_db_sync_datetime,
        )
        if assets_to_add is None:
            break

        # Remove embeddings associated with old versions of assets
        num_emb_removed = 0
        if len(asset_ids_to_remove) > 0:
            num_emb_removed = embedding_store.remove_embeddings(
                asset_ids_to_remove, collection_name
            )
            existing_doc_ids_from_past = np.array(existing_doc_ids_from_past)[
                ~np.isin(existing_doc_ids_from_past, asset_ids_to_remove)
            ].tolist()

        # Add embeddings of new assets or of new iteration of assets
        # we have just deleted
        num_emb_added = 0
        if len(assets_to_add) > 0:
            stringified_assets = [stringify_function(obj) for obj in assets_to_add]
            asset_ids = [str(obj["identifier"]) for obj in assets_to_add]

            data = [(obj, id) for obj, id in zip(stringified_assets, asset_ids)]
            loader = DataLoader(
                data, batch_size=settings.MODEL_BATCH_SIZE, num_workers=0
            )
            num_emb_added = embedding_store.store_embeddings(
                model,
                loader,
                collection_name=collection_name,
                milvus_batch_size=settings.MILVUS.BATCH_SIZE,
            )
            newly_added_doc_ids += asset_ids

        asset_collection.update(
            embeddings_added=num_emb_added, embeddings_removed=num_emb_removed
        )
        database.asset_collections.upsert(asset_collection)

        # during the traversal of AIoD assets, some of them may be deleted in between
        # which would make us skip some assets if we were to use tradinational
        # pagination without any overlap, hence the need for an overlap
        url_params.offset += int(
            settings.AIOD.WINDOW_SIZE * (1 - settings.AIOD.WINDOW_OVERLAP)
        )

    asset_collection.finish()
    database.asset_collections.upsert(asset_collection)


def get_assets_to_add_and_delete(
    url: str,
    url_params: RequestParams,
    existing_doc_ids_from_past: list[str],
    newly_added_doc_ids: list[str],
    last_db_sync_datetime: datetime | None,
) -> tuple[list[dict] | None, list[str] | None]:
    mark_recursions = []
    assets = recursive_fetch(url, url_params, mark_recursions)

    if len(assets) == 0 and len(mark_recursions) == 0:
        # We have reached the end of the AIoD database
        return None, None
    if len(assets) == 0:
        # The last page contained all but valid data
        # We need to jump to a next page
        return [], []
    if settings.AIOD.TESTING and url_params.offset >= 500:
        return None, None

    asset_ids = [str(obj["identifier"]) for obj in assets]
    modified_dates = np.array(
        [parse_asset_date(obj, none_value="now") for obj in assets]
    )

    # new assets to store that we have never encountered before
    new_asset_idx = np.where(
        ~np.isin(asset_ids, existing_doc_ids_from_past + newly_added_doc_ids)
    )[0]
    assets_to_add = [assets[idx] for idx in new_asset_idx]

    if last_db_sync_datetime is None:
        # This is executed during setup stage
        return assets_to_add, []

    # old assets that have been changed since the last time we embedded them
    # We skip assets that have just been computed (are found within newly_added_doc_ids),
    # otherwise we would have to recompute all the documents composing the pagination
    # overlap...
    # Old assets need to be deleted first, then they're stored in DB yet again
    updated_asset_idx = np.where(
        (np.isin(asset_ids, existing_doc_ids_from_past))
        & (modified_dates >= last_db_sync_datetime)
    )[0]
    asset_ids_to_del = [asset_ids[idx] for idx in updated_asset_idx]
    assets_to_add += [assets[idx] for idx in updated_asset_idx]

    return assets_to_add, asset_ids_to_del


def recursive_fetch(
    url: str, url_params: RequestParams, mark_recursions: list[int]
) -> list:
    try:
        sleep(settings.AIOD.JOB_WAIT_INBETWEEN_REQUESTS_SEC)
        queries = _build_real_url_queries(url_params)
        response = _perform_request(url, queries)
        data = response.json()
    except HTTPError as e:
        if e.response.status_code != 500:
            raise e
        if url_params.limit == 1:
            return []

        mark_recursions.append(1)
        first_half_limit = url_params.limit // 2
        second_half_limit = url_params.limit - first_half_limit

        first_half = recursive_fetch(
            url, url_params.new_page(limit=first_half_limit), mark_recursions
        )
        second_half = recursive_fetch(
            url,
            url_params.new_page(
                offset=url_params.offset + first_half_limit, limit=second_half_limit
            ),
            mark_recursions,
        )
        return first_half + second_half

    return data


def _build_real_url_queries(url_params: RequestParams) -> dict:
    return {
        "schema": "aiod",
        "offset": url_params.offset,
        "limit": url_params.limit,
        "date_modified_after": translate_datetime_to_aiod_params(url_params.from_time),
        "date_modified_before": translate_datetime_to_aiod_params(url_params.to_time),
    }
