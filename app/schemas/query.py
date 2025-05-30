from __future__ import annotations

from abc import ABC
from datetime import datetime

from pydantic import BaseModel

from app.models.filter import Filter
from app.schemas.enums import QueryStatus, SupportedAssetType, AssetTypeQueryParam
from app.schemas.search_results import SearchResults


class ReturnedAsset(BaseModel):
    asset_id: int
    asset_type: SupportedAssetType

    @staticmethod
    def create_list_from_result_set(result_set: SearchResults) -> list[ReturnedAsset]:
        return [
            ReturnedAsset(asset_id=id, asset_type=typ)
            for id, typ in zip(result_set.asset_ids, result_set.asset_types)
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
    asset_id: int
    output_asset_type: AssetTypeQueryParam
