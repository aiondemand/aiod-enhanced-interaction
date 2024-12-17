from __future__ import annotations

from app.models.condition import Condition
from app.schemas.enums import QueryStatus
from pydantic import BaseModel


class SimpleUserQueryResponse(BaseModel):
    orig_query: str
    asset_type: str
    topk: int
    status: QueryStatus = QueryStatus.QUEUED

    returned_doc_count: int = -1
    result_doc_ids: list[str] | None = None


class FilteredUserQueryResponse(SimpleUserQueryResponse):
    topic: str = ""
    filters: list[Condition] | None = None
