import gc
import logging
import threading
from functools import partial
from time import sleep
from typing import Callable

import numpy as np
import requests
import torch
from app.config import settings
from app.models.asset_collections import AssetCollection, SetupCollectionUpdate
from app.schemas.enums import AssetType
from app.services.database import Database
from app.services.embedding_store import EmbeddingStore, Milvus_EmbeddingStore
from app.services.inference.model import AiModel
from app.services.inference.text_operations import ConvertJsonToString
from requests.exceptions import ConnectTimeout, HTTPError
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
    import asyncio

    await asyncio.sleep(5)

    dev = "cuda" if first_invocation and torch.cuda.is_available() else "cpu"
    model = AiModel(dev)
    database = Database()
    embedding_store = await Milvus_EmbeddingStore.init()

    try:
        asset_types = settings.AIOD.ASSET_TYPES
        for asset_type in asset_types:
            logger.info(f"\tComputing embeddings for asset type: {asset_type.value}")

            asset_collection = database.get_asset_collection_by_type(asset_type)
            if asset_collection is None:
                # Very first invocation of this function, start setup
                asset_collection = AssetCollection(aiod_asset_type=asset_type)
                database.asset_collections.insert(asset_collection)
            elif asset_collection.last_update.finished:
                # Last update was finished, thus we create a new recurring update
                asset_collection.add_recurring_update()
                database.asset_collections.upsert(asset_collection)

            fn_kwargs = dict(
                model=model,
                stringify_function=partial(
                    ConvertJsonToString.extract_relevant_info, asset_type=asset_type
                ),
                embedding_store=embedding_store,
                database=database,
                asset_collection=asset_collection,
                asset_type=asset_type,
            )
            if asset_collection.setup_done is False:
                setup_all_aiod_assets(**fn_kwargs)
            else:
                database.asset_collections
                regular_update_aiod_assets(**fn_kwargs)
    finally:
        model.to_device("cpu")
        del model
        torch.cuda.empty_cache()
        gc.collect()


def setup_all_aiod_assets(
    model: AiModel,
    stringify_function: Callable[[dict], str],
    embedding_store: EmbeddingStore,
    database: Database,
    asset_collection: AssetCollection,
    asset_type: AssetType,
) -> None:
    asset_url = settings.AIOD.get_asset_url(asset_type)
    count_url = settings.AIOD.get_asset_count_url(asset_type)
    collection_name = settings.MILVUS.get_collection_name(asset_type)
    existing_doc_ids = embedding_store.get_all_document_ids(collection_name)

    last_update = asset_collection.last_update
    if isinstance(last_update, SetupCollectionUpdate):
        offset = last_update.aiod_asset_offset
        if offset > 0:
            logger.info(
                f"\tContinue SETUP asset embedding process with offset={offset}"
            )
    else:
        raise TypeError(
            "The last collection update is supposed to be of type 'SetupCollectionUpdate'"
        )

    while True:
        assets = recursive_fetch(asset_url, offset, settings.AIOD.WINDOW_SIZE)

        if len(assets) == 0:
            number_of_assets = _perform_request(count_url)
            if offset >= number_of_assets:
                # We have traversed all the assets
                break
            offset += settings.AIOD.WINDOW_SIZE
            continue
        elif settings.AIOD.TESTING and offset >= 500:
            # End of testing
            break

        stringified_assets = [stringify_function(obj) for obj in assets]
        asset_ids = [str(obj["identifier"]) for obj in assets]
        indices = np.where(~np.isin(asset_ids, existing_doc_ids))[0]
        if len(indices) == 0:
            continue

        data = [(stringified_assets[idx], asset_ids[idx]) for idx in indices]
        loader = DataLoader(data, batch_size=settings.MODEL_BATCH_SIZE, num_workers=0)
        new_doc_ids = embedding_store.store_embeddings(
            model,
            loader,
            collection_name=collection_name,
            milvus_batch_size=settings.MILVUS.BATCH_SIZE,
        )
        existing_doc_ids = np.hstack(
            [np.array(existing_doc_ids), np.unique(new_doc_ids)]
        ).tolist()

        asset_collection.update(assets_added=len(np.unique(new_doc_ids)))
        database.asset_collections.upsert(asset_collection)
        offset += settings.AIOD.WINDOW_SIZE

    asset_collection.finish()
    database.asset_collections.upsert(asset_collection)


def regular_update_aiod_assets(
    model: AiModel,
    stringify_function: Callable[[dict], str],
    embedding_store: EmbeddingStore,
    database: Database,
    asset_collection: AssetCollection,
    asset_type: AssetType,
) -> None:
    logger.info("Placeholder for recurring embedding function")

    # TODO
    # Retrieve all the asset changes made after specific TIMESTAMP
    # TIMESTAMP -> update.created_at time of the last update
    # this also catches changes made to assets throughout the execution of the last update

    # No matter whether we have specific assets in our vector database, we wish to
    # overwrite them with new data

    # ASSETS
    # new added assets
    # updates of the old assets

    # Update logic
    # Since the number of chunks in an updated document may be different to the original
    # number of chunks stored in the database, we need to initially delete the data
    # tied to the old documents... Having deleted documents, we may then insert them
    # once again
    # We dont do UPDATE/UPSERT

    pass


def recursive_fetch(url: str, offset: int, limit: int) -> list:
    try:
        queries = {"schema": "aiod", "offset": offset, "limit": limit}
        response = _perform_request(url, queries)
        data = response.json()
        sleep(settings.AIOD.TIMEOUT_REQUEST_INTERVAL_SEC)
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
    connection_timeout_sec: int = 60,
) -> Response:
    for _ in range(num_retries):
        try:
            response = requests.get(url, params, timeout=connection_timeout_sec)
            response.raise_for_status()
            return response
        except ConnectTimeout:
            sleep(connection_timeout_sec)

    # This exception will be only raised if we encounter
    # ConnectTimeout consecutively for multiple times
    raise ValueError("We couldn't connect to AIoD API")
