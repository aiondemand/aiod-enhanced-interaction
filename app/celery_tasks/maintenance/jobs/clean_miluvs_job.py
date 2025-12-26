import logging
from datetime import datetime

from beanie.operators import In
import numpy as np
from app import settings
from app.models import AssetForMetadataExtraction
from app.schemas.asset_id import AssetId
from app.schemas.enums import SupportedAssetType
from app.schemas.params import RequestParams
from app.services.aiod import check_aiod_asset
from app.services.embedding_store import EmbeddingStore
from app.celery_tasks.maintenance.jobs.new_embeddings_job import (
    AssetIdsAccum,
    get_assets_to_add_and_update,
)


# Achieving sufficient robustness of the embedding deletion process
# is not that important in contrast to the process of updating the assets
# Thus, there is no need to store metadata information in MongoDB regarding this
# process. You should rather view this process as some sort of garbage collector
# that is run once a month
# the immediate incorrect embeddings retrieved for a specific user query are dealt
# with using the lazy deletion approach instead
async def delete_asset_embeddings(
    embedding_store: EmbeddingStore, asset_type: SupportedAssetType, to_time: datetime
) -> None:
    all_aiod_asset_ids: list[AssetId] = []
    url_params = RequestParams(
        offset=0,
        to_time=to_time,
    )
    milvus_asset_ids = embedding_store.get_all_asset_ids(asset_type)

    # iterate over entirety of AIoD database, store all the asset IDs
    while True:
        assets_to_add, _ = get_assets_to_add_and_update(
            asset_type=asset_type,
            url_params=url_params,
            asset_ids_accum=AssetIdsAccum(),
            last_db_sync_datetime=None,
        )
        if assets_to_add is None:
            break
        all_aiod_asset_ids.extend([obj["identifier"] for obj in assets_to_add])

        # during the traversal of AIoD assets, some of them may be deleted in between
        # which would make us skip some assets if we were to use traditional
        # pagination without any overlap, hence the need for an overlap
        url_params.offset += settings.AIOD.OFFSET_INCREMENT

    # Compare AIoD assets to Milvus assets, the set diff assets
    # are candidates for deletion; each candidate is explicitly checked against AIoD
    candidates_idx = np.where(~np.isin(milvus_asset_ids, all_aiod_asset_ids))[0]
    candidates_for_del = [milvus_asset_ids[idx] for idx in candidates_idx]

    if len(candidates_for_del) > 0:
        logging.info(
            f"\t\t[RECURRING MILVUS DELETE] {len(candidates_for_del)} assets ({asset_type.value}) have been chosen as candidates for deletion."
        )

    ids_to_really_delete = [
        id
        for id in candidates_for_del
        if check_aiod_asset(
            id, asset_type, sleep_time=settings.AIOD.JOB_WAIT_INBETWEEN_REQUESTS_SEC
        )
        is False
    ]
    if len(ids_to_really_delete) > 0:
        embedding_store.remove_embeddings(ids_to_really_delete, asset_type)

        # Remove assets from MongoDB if they exist (AssetForMetadataExtraction collection)
        if settings.extracts_metadata_from_asset(asset_type):
            await AssetForMetadataExtraction.delete_docs(
                In(AssetForMetadataExtraction.asset_id, ids_to_really_delete)
            )

        logging.info(
            f"\t\t[RECURRING MILVUS DELETE] {len(ids_to_really_delete)} assets ({asset_type.value}) have been deleted from the Milvus database."
        )
