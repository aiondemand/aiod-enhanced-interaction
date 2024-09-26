from __future__ import annotations

from app.schemas.enums import QueryStatus
from pydantic import BaseModel


class UserQueryResponse(BaseModel):
    status: QueryStatus = QueryStatus.QUEUED
    result_doc_ids: list[str] | None = None
