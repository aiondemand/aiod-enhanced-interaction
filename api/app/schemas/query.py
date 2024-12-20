from __future__ import annotations

from app.models.filter import Filter
from app.schemas.enums import QueryStatus
from pydantic import BaseModel


class BaseUserQueryResponse(BaseModel):
    orig_query: str
    asset_type: str
    topk: int
    status: QueryStatus = QueryStatus.QUEUED

    returned_doc_count: int = -1
    result_doc_ids: list[str] | None = None


class SimpleUserQueryResponse(BaseUserQueryResponse):
    pass


class FilteredUserQueryResponse(BaseUserQueryResponse):
    topic: str = ""
    filters: list[Filter] | None = None
