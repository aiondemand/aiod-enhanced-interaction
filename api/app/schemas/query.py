from __future__ import annotations

from app.models.filter import Filter
from app.schemas.enums import QueryStatus
from pydantic import BaseModel


class BaseUserQueryResponse(BaseModel):
    search_query: str

    asset_type: str
    status: QueryStatus = QueryStatus.QUEUED
    offset: int
    limit: int
    num_hits: int = -1

    returned_doc_count: int = -1
    result_doc_ids: list[str] | None = None
    result_docs: list[dict] | None = None


class SimpleUserQueryResponse(BaseUserQueryResponse):
    pass


class FilteredUserQueryResponse(BaseUserQueryResponse):
    topic: str = ""
    filters: list[Filter] | None = None
