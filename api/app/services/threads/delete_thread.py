import logging
import threading
from datetime import datetime, timezone
from time import sleep

import numpy as np
from app.config import settings
from app.helper import _perform_request
from app.schemas.enums import AssetType
from app.schemas.request_params import RequestParams
from app.services.embedding_store import EmbeddingStore, Milvus_EmbeddingStore
from app.services.threads.embedding_thread import get_assets_to_add_and_delete
from requests.exceptions import HTTPError

job_lock = threading.Lock()


async def delete_embeddings_of_aiod_assets_wrapper() -> None:
    # Achieving sufficient robustness of the embedding deletion process
    # is not that important in contrast to the process of updating the assets
    # Thus, theres no need to store metadata information in tinyDB regarding this
    # process. You should rather view this process as some sort of garbage collector
    # that is run once a month
    # the immediate incorrect embeddings retrieved for a specific user query are dealt
    # with using the lazy deletion approach instead
    if job_lock.acquire(blocking=False):
        try:
            logging.info(
                "[RECURRING DELETE] Scheduled task for deleting asset embeddings has started."
            )

            embedding_store = await Milvus_EmbeddingStore.init()
            to_time = datetime.now(tz=timezone.utc)

            for asset_type in settings.AIOD.ASSET_TYPES:
                logging.info(f"\tDeleting embeddings of asset type: {asset_type.value}")
                delete_asset_embeddings(embedding_store, asset_type, to_time=to_time)

            logging.info(
                "[RECURRING DELETE] Scheduled task for deleting asset embeddings has ended."
            )
        finally:
            job_lock.release()
    else:
        logging.info(
            "Scheduled task for deleting skipped (previous task is still running)"
        )


def delete_asset_embeddings(
    embedding_store: EmbeddingStore, asset_type: AssetType, to_time: datetime
) -> None:
    all_aiod_doc_ids = []
    url_params = RequestParams(
        offset=0,
        limit=settings.AIOD.WINDOW_SIZE,
        to_time=to_time,
    )
    milvus_doc_ids = embedding_store.get_all_document_ids(asset_type)

    # iterate over entirety of AIoD database, store all the doc IDs
    while True:
        assets_to_add, _ = get_assets_to_add_and_delete(
            url=settings.AIOD.get_assets_url(asset_type),
            url_params=url_params,
            existing_doc_ids_from_past=[],
            newly_added_doc_ids=[],
            last_db_sync_datetime=None,
        )
        if assets_to_add is None:
            break
        all_aiod_doc_ids.extend([str(obj["identifier"]) for obj in assets_to_add])

        # during the traversal of AIoD assets, some of them may be deleted in between
        # which would make us skip some assets if we were to use tradinational
        # pagination without any overlap, hence the need for an overlap
        url_params.offset += settings.AIOD.OFFSET_INCREMENT

    # Compare AIoD assets to Milvus assets, the set diff assets
    # are candidates for deletion; each candidate is explicitly checked against AIoD
    candidates_idx = np.where(~np.isin(milvus_doc_ids, all_aiod_doc_ids))[0]
    candidates_for_del = [milvus_doc_ids[idx] for idx in candidates_idx]

    if len(candidates_for_del) > 0:
        logging.info(
            f"\t{len(candidates_for_del)} assets ({asset_type.value}) have been chosen as candidates for deletion."
        )

    ids_to_really_delete = [
        id
        for id in candidates_for_del
        if check_document_existence(
            id, asset_type, sleep_time=settings.AIOD.JOB_WAIT_INBETWEEN_REQUESTS_SEC
        )
        is False
    ]
    if len(ids_to_really_delete) > 0:
        embedding_store.remove_embeddings(ids_to_really_delete, asset_type)
        logging.info(
            f"\t{len(ids_to_really_delete)} assets ({asset_type.value}) have been deleted from the Milvus database."
        )


def check_document_existence(
    doc_id: str, asset_type: AssetType, sleep_time: float = 0.1
) -> bool:
    try:
        sleep(sleep_time)
        _perform_request(settings.AIOD.get_asset_by_id_url(doc_id, asset_type))
        return True
    except HTTPError as e:
        if e.response.status_code == 404:
            return False
        raise
