from __future__ import annotations

from datetime import datetime, timezone
from functools import partial
from uuid import uuid4

from app.models.condition import Condition
from app.schemas.enums import AssetType, QueryStatus
from app.schemas.query import FilteredUserQueryResponse, SimpleUserQueryResponse
from app.schemas.search_results import SearchResults
from pydantic import BaseModel, Field


class UserQuery(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    orig_query: str

    asset_type: AssetType
    topk: int
    status: QueryStatus = QueryStatus.QUEUED

    created_at: datetime = Field(default_factory=partial(datetime.now, tz=timezone.utc))
    updated_at: datetime = Field(default_factory=partial(datetime.now, tz=timezone.utc))
    result_set: SearchResults | None = None

    apply_filtering: bool
    topic: str = ""
    filters: list[Condition] | None = None

    def update_status(self, status: QueryStatus) -> None:
        self.status = status
        self.updated_at = datetime.now(tz=timezone.utc)

    # TODO maybe a function for updating filters/topic/result_set ???

    def map_to_response(self) -> SimpleUserQueryResponse:
        response_class = (
            FilteredUserQueryResponse
            if self.apply_filtering
            else SimpleUserQueryResponse
        )
        if self.result_set is None:
            return response_class(**self.model_dump())

        doc_ids = self.result_set.doc_ids
        return response_class(
            returned_doc_count=len(doc_ids), result_doc_ids=doc_ids, **self.model_dump()
        )

    @staticmethod
    def sort_function_to_populate_queue(query):
        return (query.status != QueryStatus.IN_PROGESS, query.updated_at.timestamp())
