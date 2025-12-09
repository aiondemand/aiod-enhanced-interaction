from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.config import settings
from app.schemas.asset_id import AssetId
from app.schemas.enums import SupportedAssetType


class RequestParams(BaseModel):
    offset: int = 0
    limit: int = settings.AIOD.WINDOW_SIZE
    from_time: datetime | None = None
    to_time: datetime | None = None
    direction: Literal["asc", "desc"] = "asc"
    sort: Literal["date_created", "date_modified"] = "date_modified"

    def new_page(self, offset: int | None = None, limit: int | None = None) -> RequestParams:
        new_obj = RequestParams(**self.model_dump())

        if offset is not None:
            new_obj.offset = offset
        if limit is not None:
            new_obj.limit = limit
        return new_obj


class VectorSearchParams(BaseModel, ABC):
    data: list[list[float]]
    topk: int
    asset_type: SupportedAssetType

    # filter related attributes
    metadata_filter: str = ""
    asset_ids_to_exclude: list[AssetId] = Field(default_factory=list)

    @abstractmethod
    def get_params(self) -> dict:
        raise NotImplementedError


class MilvusSearchParams(VectorSearchParams):
    group_by_field: str = "asset_id"
    output_fields: list[str] = Field(default_factory=lambda: ["asset_id"])
    search_params: dict = Field(default_factory=lambda: {"metric_type": "COSINE"})

    @property
    def filter(self) -> str:
        if len(self.asset_ids_to_exclude) > 0 and len(self.metadata_filter) > 0:
            return f"({self.metadata_filter}) and (asset_id not in {self.asset_ids_to_exclude})"
        elif len(self.asset_ids_to_exclude) > 0:
            return f"asset_id not in {self.asset_ids_to_exclude}"
        elif len(self.metadata_filter) > 0:
            return self.metadata_filter
        else:
            return ""

    def get_params(self) -> dict:
        return {
            "data": self.data,
            "limit": self.topk,
            "group_by_field": self.group_by_field,
            "output_fields": self.output_fields,
            "search_params": self.search_params,
            "filter": self.filter,
        }
