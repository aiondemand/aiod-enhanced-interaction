from __future__ import annotations

from datetime import datetime
from typing import TypeAlias

from app.models.filter import Filter
from app.schemas.enums import QueryStatus
from pydantic import BaseModel


class BaseUserQueryResponse(BaseModel):
    search_query: str
    asset_type: str
    status: QueryStatus = QueryStatus.QUEUED
    # offset: int
    # limit: int
    topk: int


class BaseUserQueryResponseWithResults(BaseUserQueryResponse):
    # num_hits: int = -1
    expires_at: datetime | None = None
    returned_doc_count: int = -1
    result_doc_ids: list[str] | None = None
    result_docs: list[dict] | None = None


class SimpleUserQueryResponse(BaseUserQueryResponse):
    pass


class SimpleUserQueryResponseWithResults(BaseUserQueryResponseWithResults):
    pass


class FilteredUserQueryResponse(BaseUserQueryResponse):
    pass


class ManualFilteredUserQueryResponse(BaseUserQueryResponse):
    topic: str = ""
    filters: list[Filter] | None = None


class FilteredUserQueryResponseWithResults(
    ManualFilteredUserQueryResponse, BaseUserQueryResponseWithResults
):
    pass


SimpleUserQueryResponseType: TypeAlias = (
    SimpleUserQueryResponse | SimpleUserQueryResponseWithResults
)
FilteredUserQueryResponseType: TypeAlias = (
    FilteredUserQueryResponse
    | ManualFilteredUserQueryResponse
    | FilteredUserQueryResponseWithResults
)
