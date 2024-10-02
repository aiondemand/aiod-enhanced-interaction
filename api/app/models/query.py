from __future__ import annotations

from datetime import datetime, timezone
from functools import partial
from uuid import uuid4

from app.schemas.enums import AssetType, QueryStatus
from app.schemas.query import UserQueryResponse
from app.schemas.search_results import SearchResults
from pydantic import BaseModel, Field


class UserQuery(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    query: str
    asset_type: AssetType
    topk: int
    status: QueryStatus = QueryStatus.QUEUED
    created_at: datetime = Field(default_factory=partial(datetime.now, tz=timezone.utc))
    updated_at: datetime = Field(default_factory=partial(datetime.now, tz=timezone.utc))
    result_set: SearchResults | None = None

    def update_status(self, status: QueryStatus) -> None:
        self.status = status
        self.updated_at = datetime.now(tz=timezone.utc)

    def map_to_response(self) -> UserQueryResponse:
        doc_ids = None
        if self.result_set is not None:
            doc_ids = self.result_set.doc_ids

        return UserQueryResponse(
            status=self.status, num_doc_ids=len(doc_ids), result_doc_ids=doc_ids
        )

    @staticmethod
    def sort_function_to_populate_queue(query):
        return (query.status != QueryStatus.IN_PROGESS, query.updated_at.timestamp())
