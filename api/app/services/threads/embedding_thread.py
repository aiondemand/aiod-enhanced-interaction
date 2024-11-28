import gc
import logging
import threading
from datetime import datetime, timezone
from functools import partial
from time import sleep
from typing import Callable

import numpy as np
import requests
import torch
from app.config import settings
from app.helper import translate_datetime_to_aiod_params
from app.models.asset_collections import AssetCollection
from app.schemas.enums import AssetType
from app.schemas.request_params import RequestParams
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
    model = AiModel(device=AiModel.get_device())
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
    asset_url = settings.AIOD.get_asset_url(asset_type)
    collection_name = settings.MILVUS.get_collection_name(asset_type)
    existing_doc_ids = embedding_store.get_all_document_ids(collection_name)

    last_update = asset_collection.last_update
    url_params = RequestParams(
        offset=last_update.aiod_asset_offset,
        limit=settings.AIOD.WINDOW_SIZE,
        from_time=getattr(
            last_update, "from_time", datetime.fromtimestamp(0, tz=timezone.utc)
        ),
        to_time=last_update.to_time,
    )
    if url_params.offset > 0:
        logger.info(
            f"\tContinue asset embedding process from asset offset={url_params.offset}"
        )

    while True:
        assets_to_add, asset_ids_to_remove = get_assets_to_add_and_delete(
            asset_url,
            url_params,
            existing_doc_ids=existing_doc_ids,
            setup_stage=is_setup_stage,
        )
        if assets_to_add is None:
            break

        # Remove embeddings associated with old versions of assets
        num_emb_removed = 0
        if len(asset_ids_to_remove) > 0:
            num_emb_removed = embedding_store.remove_embeddings(
                asset_ids_to_remove, collection_name
            )
            existing_doc_ids = np.array(existing_doc_ids)[
                ~np.isin(existing_doc_ids, asset_ids_to_remove)
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
            existing_doc_ids += asset_ids

        asset_collection.update(
            embeddings_added=num_emb_added, embeddings_removed=num_emb_removed
        )
        database.asset_collections.upsert(asset_collection)
        url_params.offset += settings.AIOD.WINDOW_SIZE

    asset_collection.finish()
    database.asset_collections.upsert(asset_collection)


def get_assets_to_add_and_delete(
    url: str,
    url_params: RequestParams,
    existing_doc_ids: list[str],
    setup_stage: bool = False,
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
    new_asset_idx = np.where(~np.isin(asset_ids, existing_doc_ids))[0]
    existing_asset_idx = np.where(np.isin(asset_ids, existing_doc_ids))[0]

    assets_to_add = [assets[idx] for idx in new_asset_idx]
    if setup_stage:
        # In setup stage, we dont delete any assets despite potentionally already
        # being in the vector db
        asset_ids_to_del = []
        return assets_to_add, asset_ids_to_del
    else:
        # When performing recurring updates, we do delete existing assets as these
        # ones will be their new iteration
        assets_to_add.extend([assets[idx] for idx in existing_asset_idx])
        asset_ids_to_del = [asset_ids[idx] for idx in existing_asset_idx]
        return assets_to_add, asset_ids_to_del


def recursive_fetch(
    url: str, url_params: RequestParams, mark_recursions: list[int]
) -> list:
    try:
        sleep(settings.AIOD.TIMEOUT_REQUEST_INTERVAL_SEC)
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

        first_half = recursive_fetch(url, url_params.new_page(limit=first_half_limit))
        second_half = recursive_fetch(
            url,
            url_params.new_page(
                offset=url_params.offset + first_half_limit, limit=second_half_limit
            ),
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
