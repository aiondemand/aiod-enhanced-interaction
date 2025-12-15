from __future__ import annotations

import json
from typing import Annotated

from beanie import Indexed


from app.schemas.asset_id import AssetId
from app.models.mongo import BaseDatabaseEntity, MongoDocument
from app.schemas.enums import SupportedAssetType


# Regardless of whether we have metadata extraction turned on or off, we should still
# keep track of all the assets we wish to extract metadata from, so that if after some 
# time we actually turn on the medata extraction logic, we wouldnt need to do any additional migration logic
class AssetForMetadataExtraction(MongoDocument, BaseDatabaseEntity):
    asset_id: Annotated[AssetId, Indexed()]
    asset_type: SupportedAssetType
    asset_json_str: str

    # We check whether the version found in the MongoDB is the same as the last version found in the Milvus
    asset_version: int

    class Settings:
        name = "assetsForMetadataExtraction"

    @classmethod
    def create_asset(
        cls, asset: dict, asset_type: SupportedAssetType, asset_version: int = 0
    ) -> AssetForMetadataExtraction:
        return cls(
            asset_id=asset["identifier"],
            asset_json_str=json.dumps(asset),
            asset_type=asset_type,
            asset_version=asset_version,
        )

    @property
    def asset(self) -> dict:
        return json.loads(self.asset_json_str)
