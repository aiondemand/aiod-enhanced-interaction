from __future__ import annotations

from app.models.filter import Filter
from app.schemas.enums import QueryStatus
from pydantic import BaseModel


class BaseUserQueryResponse(BaseModel):
    asset_type: str
    status: QueryStatus = QueryStatus.QUEUED
    topk: int
    returned_doc_count: int = -1
    result_doc_ids: list[str] | None = None


class SimpleUserQueryResponse(BaseUserQueryResponse):
    search_query: str


class FilteredUserQueryResponse(BaseUserQueryResponse):
    search_query: str
    filters: list[Filter] | None = None


class SimilarUserQueryResponse(BaseUserQueryResponse):
    asset_id: int
