from __future__ import annotations

import json

from beanie import Indexed


from app.schemas.asset_id import AssetId
from app.models.mongo import MongoDocument
from app.schemas.enums import SupportedAssetType


class AssetForMetadataExtraction(MongoDocument):
    asset_id: Indexed[AssetId]
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
