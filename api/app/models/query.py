from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Type
from uuid import uuid4

from app.models.filter import Filter
from app.schemas.enums import AssetType, QueryStatus
from app.schemas.query import (
    FilteredUserQueryResponse,
    FilteredUserQueryResponseType,
    FilteredUserQueryResponseWithResults,
    ManualFilteredUserQueryResponse,
    SimpleUserQueryResponse,
    SimpleUserQueryResponseType,
    SimpleUserQueryResponseWithResults,
)
from app.schemas.search_results import SearchResults
from pydantic import BaseModel, Field


class BaseUserQuery(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    search_query: str

    asset_type: AssetType
    status: QueryStatus = QueryStatus.QUEUED

    # Offset is only approximate since it is applied on Milvus embeddings rather
    # than on the actual documents returned by our service, and since one document may
    # have multiple embeddings, the resulting offset may not be accurate and there
    # may be some overlap between pages.
    offset: int
    limit: int
    return_assets: bool = False

    created_at: datetime = Field(default_factory=partial(datetime.now, tz=timezone.utc))
    updated_at: datetime = Field(default_factory=partial(datetime.now, tz=timezone.utc))
    expires_at: datetime | None = None
    result_set: SearchResults | None = None

    def update_status(self, status: QueryStatus) -> None:
        self.status = status
        self.updated_at = datetime.now(tz=timezone.utc)

        if status in (QueryStatus.COMPLETED, QueryStatus.FAILED):
            # an arbitrarily chosen expiration date
            self.expires_at = datetime.now(tz=timezone.utc) + timedelta(hours=1)

    def map_to_response(self, response_model: Type[BaseModel]) -> BaseModel:
        # Rename limit to topk for now
        kwargs = self.model_dump()
        kwargs["topk"] = kwargs.pop("limit")

        if self.status != QueryStatus.COMPLETED:
            return response_model(**kwargs)

        doc_ids = self.result_set.doc_ids
        docs = self.result_set.documents if self.return_assets else None
        return response_model(
            returned_doc_count=len(doc_ids),
            result_doc_ids=doc_ids,
            result_docs=docs,
            # num_hits=self.result_set.num_hits,
            **kwargs,
        )

    @staticmethod
    def sort_function_to_populate_queue(query):
        return (query.status != QueryStatus.IN_PROGRESS, query.updated_at.timestamp())


class SimpleUserQuery(BaseUserQuery):
    def map_to_response(self) -> SimpleUserQueryResponseType:
        return super().map_to_response(
            SimpleUserQueryResponseWithResults
            if self.status == QueryStatus.COMPLETED
            else SimpleUserQueryResponse
        )


class FilteredUserQuery(BaseUserQuery):
    topic: str = ""
    filters: list[Filter] | None = None

    @property
    def invoke_llm_for_parsing(self) -> bool:
        return self.filters is None

    def update_query_metadata(self, topic: str, filters: list[Filter]) -> None:
        self.topic = topic
        self.filters = filters

    def map_to_response(self) -> FilteredUserQueryResponseType:
        if self.filters is None and self.status != QueryStatus.COMPLETED:
            response_model = FilteredUserQueryResponse
        elif self.status == QueryStatus.COMPLETED:
            response_model = FilteredUserQueryResponseWithResults
        else:
            response_model = ManualFilteredUserQueryResponse

        return super().map_to_response(response_model)
