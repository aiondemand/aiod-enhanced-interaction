import gc
import logging
import threading
from datetime import datetime
from functools import partial
from time import sleep
from typing import Callable

import numpy as np
import requests
import torch
from app.config import settings
from app.helper import parse_asset_date
from app.models.asset_collections import AssetCollection
from app.schemas.enums import AssetType
from app.services.database import Database
from app.services.embedding_store import EmbeddingStore, Milvus_EmbeddingStore
from app.services.inference.model import AiModel
from app.services.inference.text_operations import ConvertJsonToString
from requests.exceptions import HTTPError, Timeout
from requests.models import Response
from torch.utils.data import DataLoader

job_lock = threading.Lock()
logger = logging.getLogger("uvicorn")


async def compute_embeddings_for_aiod_assets_wrapper(
    first_invocation: bool = False,
) -> None:
    if job_lock.acquire(blocking=False):
        try:
            log_msg = (
                "[STARTUP] Initial task for computing asset embeddings has started"
                if first_invocation
                else "Scheduled task for computing asset embeddings has started"
            )
            logger.info(log_msg)
            await compute_embeddings_for_aiod_assets(first_invocation)
            logger.info("Scheduled task for computing asset embeddings has ended.")
        finally:
            job_lock.release()
    else:
        logger.info("Scheduled task skipped (previous task is still running)")


async def compute_embeddings_for_aiod_assets(first_invocation: bool) -> None:
    model = AiModel(device=AiModel.get_device(first_invocation))
    database = Database()
    embedding_store = await Milvus_EmbeddingStore.init()

    try:
        asset_types = settings.AIOD.ASSET_TYPES
        for asset_type in asset_types:
            logger.info(f"\tComputing embeddings for asset type: {asset_type.value}")

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
    is_setup_stage = asset_collection.setup_done is False
    asset_url = settings.AIOD.get_asset_url(asset_type, is_setup_stage)
    count_url = settings.AIOD.get_asset_count_url(asset_type, is_setup_stage)
    collection_name = settings.MILVUS.get_collection_name(asset_type)
    existing_doc_ids = embedding_store.get_all_document_ids(collection_name)

    last_update = asset_collection.last_update
    offset = last_update.aiod_asset_offset
    if offset > 0:
        logger.info(f"\tContinue asset embedding process from asset offset={offset}")

    while True:
        assets_to_add, asset_ids_to_remove = get_assets_to_add_and_delete(
            asset_url,
            offset,
            existing_doc_ids=existing_doc_ids,
            count_url=count_url,
            setup_stage=is_setup_stage,
            from_time=getattr(last_update, "from_time", None),
            to_time=last_update.to_time,
        )
        if assets_to_add is None:
            break

        # Remove embeddings
        num_emb_removed = 0
        if len(asset_ids_to_remove) > 0:
            num_emb_removed = embedding_store.remove_embeddings(
                asset_ids_to_remove, collection_name
            )
            existing_doc_ids = np.array(existing_doc_ids)[
                ~np.isin(existing_doc_ids, asset_ids_to_remove)
            ].tolist()

        # Add embeddings
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
            existing_doc_ids += asset_ids

        asset_collection.update(
            embeddings_added=num_emb_added, embeddings_removed=num_emb_removed
        )
        database.asset_collections.upsert(asset_collection)
        offset += settings.AIOD.WINDOW_SIZE

    asset_collection.finish()
    database.asset_collections.upsert(asset_collection)


def get_assets_to_add_and_delete(
    url: str,
    offset: int,
    existing_doc_ids: list[str],
    count_url: str | None = None,
    setup_stage: bool = False,
    from_time: datetime | None = None,
    to_time: datetime | None = None,
) -> tuple[list[dict] | None, list[str] | None]:
    assets = recursive_fetch(url, offset, settings.AIOD.WINDOW_SIZE)

    if len(assets) == 0:
        if count_url is None:
            return None, None
        total_number_of_assets = _perform_request(count_url).json()
        if offset >= total_number_of_assets:
            return None, None
        return [], []
    elif settings.AIOD.TESTING and offset >= 500:
        return None, None

    # TODO this function will be changed once we have a new AIoD endpoints
    # that work with different schemas etc...

    # Only assets updated till the start of this job
    assets = [
        obj for obj in assets if (parse_asset_date(obj, none_value="zero") < to_time)
    ]
    asset_ids = [str(obj["identifier"]) for obj in assets]
    new_indices = np.where(~np.isin(asset_ids, existing_doc_ids))[0]
    assets_to_add = [assets[idx] for idx in new_indices]

    if setup_stage:
        asset_ids_to_del = []
        return assets_to_add, asset_ids_to_del
    else:
        modified_dates = np.array(
            [parse_asset_date(obj, none_value="now") for obj in assets]
        )
        update_indices = np.where(
            (np.isin(asset_ids, existing_doc_ids)) & (modified_dates > from_time)
        )[0]

        assets_to_add.extend([assets[idx] for idx in update_indices])
        asset_ids_to_del = [asset_ids[idx] for idx in update_indices]
        return assets_to_add, asset_ids_to_del


def recursive_fetch(url: str, offset: int, limit: int) -> list:
    try:
        sleep(settings.AIOD.TIMEOUT_REQUEST_INTERVAL_SEC)
        queries = {"schema": "aiod", "offset": offset, "limit": limit}
        response = _perform_request(url, queries)
        data = response.json()
    except HTTPError as e:
        if e.response.status_code != 500:
            raise e
        if limit == 1:
            return []

        first_half_limit = limit // 2
        second_half_limit = limit - first_half_limit

        first_half = recursive_fetch(url, offset, first_half_limit)
        second_half = recursive_fetch(url, offset + first_half_limit, second_half_limit)
        return first_half + second_half

    return data


def _perform_request(
    url: str,
    params: dict | None = None,
    num_retries: int = 3,
    connection_timeout_sec: int = 30,
) -> Response:
    # timeout based on the size of the requested response
    # 1000 assets == additional 1 minute of timeout
    limit = 0 if params is None else params.get("limit", 0)
    request_timeout = connection_timeout_sec + int(limit * 0.06)

    for _ in range(num_retries):
        try:
            response = requests.get(url, params, timeout=request_timeout)
            response.raise_for_status()
            return response
        except Timeout:
            logging.warning("AIoD endpoints are unresponsive. Retrying...")
            sleep(connection_timeout_sec)

    # This exception will be only raised if we encounter exception
    # Timeout (ReadTimeout / ConnectionTimeout) consecutively for multiple times
    err_msg = "We couldn't connect to AIoD API"
    logging.error(err_msg)
    raise ValueError(err_msg)
