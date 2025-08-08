from __future__ import annotations

from abc import ABC
from datetime import datetime

from pydantic import BaseModel

from app.models.filter import Filter
from app.schemas.asset_id import AssetId
from app.schemas.enums import QueryStatus, SupportedAssetType, AssetTypeQueryParam
from app.schemas.search_results import AssetResults


class ReturnedAsset(BaseModel):
    asset_id: AssetId
    asset_type: SupportedAssetType
    asset: dict | None = None

    @staticmethod
    def create_list_from_asset_results(
        asset_results: AssetResults, return_entire_assets: bool = False
    ) -> list[ReturnedAsset]:
        return [
            ReturnedAsset(
                asset_id=id, asset_type=typ, asset=asset if return_entire_assets else None
            )
            for id, typ, asset in zip(
                asset_results.asset_ids, asset_results.asset_types, asset_results.assets
            )
        ]


class BaseUserQueryResponse(BaseModel, ABC):
    status: QueryStatus = QueryStatus.QUEUED
    topk: int
    results: list[ReturnedAsset] | None = None
    expires_at: datetime | None = None


class SimpleUserQueryResponse(BaseUserQueryResponse):
    asset_type: AssetTypeQueryParam
    search_query: str


class FilteredUserQueryResponse(BaseUserQueryResponse):
    asset_type: SupportedAssetType
    search_query: str
    filters: list[Filter] | None = None


class RecommenderUserQueryResponse(BaseUserQueryResponse):
    asset_type: SupportedAssetType
    asset_id: AssetId
    output_asset_type: AssetTypeQueryParam
