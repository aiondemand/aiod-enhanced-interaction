from __future__ import annotations

from abc import ABC
from datetime import datetime

from pydantic import BaseModel

from app.models.filter import Filter
from app.schemas.enums import QueryStatus


class BaseUserQueryResponse(BaseModel, ABC):
    asset_type: str
    status: QueryStatus = QueryStatus.QUEUED
    topk: int
    returned_asset_count: int = -1
    result_asset_ids: list[int] | None = None
    result_assets: list[dict] | None = None
    expires_at: datetime | None = None


class SimpleUserQueryResponse(BaseUserQueryResponse):
    search_query: str


class FilteredUserQueryResponse(BaseUserQueryResponse):
    search_query: str
    filters: list[Filter] | None = None


class RecommenderUserQueryResponse(BaseUserQueryResponse):
    asset_id: int
    output_asset_type: str
