from __future__ import annotations

import json


from app.schemas.asset_id import AssetId
from app.models.mongo import MongoDocument
from app.schemas.enums import SupportedAssetType


class AssetForMetadataExtraction(MongoDocument):
    # TODO create index on top of the asset_id
    asset_id: AssetId
    asset_type: SupportedAssetType
    asset_json_str: str

    # We check whether the version found in the MongoDB is the same as the last version found in the Milvus
    version: int

    @classmethod
    def create_asset(
        cls, asset: dict, asset_type: SupportedAssetType, version: int = 0
    ) -> AssetForMetadataExtraction:
        return cls(
            asset_id=asset["identifier"],
            asset_json_str=json.dumps(asset),
            asset_type=asset_type,
            version_number=version,
        )

    @property
    def asset(self) -> dict:
        return json.loads(self.asset_json_str)
