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
from app.schemas.enums import AssetType
from app.services.embedding_store import EmbeddingStore, Milvus_EmbeddingStore
from app.services.inference.model import AiModel
from app.services.inference.text_operations import ConvertJsonToString
from requests.exceptions import ConnectTimeout, HTTPError
from requests.models import Response
from torch.utils.data import DataLoader

job_lock = threading.Lock()
logger = logging.getLogger("uvicorn")


def compute_embeddings_for_AIoD_assets_wrapper(first_invocation: bool = False):
    if job_lock.acquire(blocking=False):
        try:
            log_msg = (
                "[STARTUP] Initial task for computing asset embeddings has started"
                if first_invocation
                else "Scheduled task for computing asset embeddings has started"
            )
            logger.info(log_msg)
            compute_embeddings_for_AIoD_assets(first_invocation)
            logger.info("Scheduled task for computing asset embeddings has ended.")
        finally:
            job_lock.release()
    else:
        logger.info("Scheduled task skipped (previous task is still running)")


def compute_embeddings_for_AIoD_assets(first_invocation: bool):
    dev = "cuda" if first_invocation and torch.cuda.is_available() else "cpu"
    model = AiModel(dev)
    embedding_store = Milvus_EmbeddingStore()

    try:
        asset_types = settings.AIOD.ASSET_TYPES
        for asset_type in asset_types:
            logger.info(f"\t Computing embeddings for asset type: {asset_type.value}")
            stringify_function = partial(
                ConvertJsonToString.extract_relevant_info, asset_type=asset_type
            )
            embed_aiod_assets(
                model, stringify_function, embedding_store, asset_type=asset_type
            )
    finally:
        model.to_device("cpu")
        del model
        torch.cuda.empty_cache()
        gc.collect()


def embed_aiod_assets(
    model: AiModel,
    stringify_function: Callable[[dict], str],
    embedding_store: EmbeddingStore,
    asset_type: AssetType,
) -> None:
    asset_url = settings.AIOD.get_asset_url(asset_type)
    count_url = settings.AIOD.get_asset_count_url(asset_type)
    collection_name = settings.MILVUS.get_collection_name(asset_type)
    offset = 0

    while True:
        assets = recursive_fetch(asset_url, offset, settings.AIOD.WINDOW_SIZE)
        if len(assets) > 0:
            stringified_assets = [stringify_function(obj) for obj in assets]
            asset_ids = [str(obj["identifier"]) for obj in assets]
            existing_doc_ids = embedding_store.get_all_document_ids(collection_name)

            indices = np.where(~np.isin(asset_ids, existing_doc_ids))[0]
            if len(indices) == 0:
                continue

            data = [(stringified_assets[idx], asset_ids[idx]) for idx in indices]
            loader = DataLoader(
                data, batch_size=settings.MODEL_BATCH_SIZE, num_workers=0
            )

            embedding_store.store_embeddings(
                model,
                loader,
                collection_name=collection_name,
                milvus_batch_size=settings.MILVUS.BATCH_SIZE,
            )
        else:
            number_of_assets = _perform_request(count_url)
            if offset >= number_of_assets:
                # We have traversed all the assets
                break

        offset += settings.AIOD.WINDOW_SIZE


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
