from __future__ import annotations

from abc import abstractmethod
from datetime import datetime, timezone
from functools import partial
from typing import Type
from uuid import uuid4

from app.models.filter import Filter
from app.schemas.enums import AssetType, QueryStatus
from app.schemas.query import (
    BaseUserQueryResponse,
    FilteredUserQueryResponse,
    SimpleUserQueryResponse,
    SimilarQueryResponse
)
from app.schemas.search_results import SearchResults
from pydantic import BaseModel, Field


class BaseUserQuery(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    search_query: str

    asset_type: AssetType
    topk: int
    status: QueryStatus = QueryStatus.QUEUED
    topk: int

    created_at: datetime = Field(default_factory=partial(datetime.now, tz=timezone.utc))
    updated_at: datetime = Field(default_factory=partial(datetime.now, tz=timezone.utc))
    result_set: SearchResults | None = None

    def update_status(self, status: QueryStatus) -> None:
        self.status = status
        self.updated_at = datetime.now(tz=timezone.utc)

    def _map_to_response(
        self, response_model: Type[BaseUserQueryResponse]
    ) -> BaseUserQueryResponse:
        if self.status != QueryStatus.COMPLETED:
            return response_model(**self.model_dump())

        doc_ids = self.result_set.doc_ids
        return response_model(
            returned_doc_count=len(doc_ids),
            result_doc_ids=doc_ids,
            **self.model_dump(),
        )

    @staticmethod
    def sort_function_to_populate_queue(query):
        return (query.status != QueryStatus.IN_PROGRESS, query.updated_at.timestamp())

    @abstractmethod
    def map_to_response(self) -> BaseUserQueryResponse:
        raise NotImplementedError


class SimpleUserQuery(BaseUserQuery):
    def map_to_response(self) -> SimpleUserQueryResponse:
        return self._map_to_response(SimpleUserQueryResponse)


class FilteredUserQuery(BaseUserQuery):
    filters: list[Filter] | None = None

    @property
    def invoke_llm_for_parsing(self) -> bool:
        return self.filters is None

    def map_to_response(self) -> FilteredUserQueryResponse:
        return self._map_to_response(FilteredUserQueryResponse)


class SimilarQuery(BaseModel):
    asset_id: str
    status: QueryStatus = QueryStatus.QUEUED
    updated_at: datetime = Field(default_factory=partial(datetime.now, tz=timezone.utc))
    result_set: SearchResults | None = None

    def map_to_response(self) -> SimilarQueryResponse:
        if self.result_set is None:
            return SimilarQueryResponse(
                status=self.status,
                asset_id=self.asset_id,
            )

        doc_ids = self.result_set.doc_ids
        return SimilarQueryResponse(
            status=self.status,
            asset_id=self.asset_id,
            num_doc_ids=len(doc_ids),
            result_doc_ids=doc_ids,
        )

    def update_status(self, status: QueryStatus) -> None:
        self.status = status
        self.updated_at = datetime.now(tz=timezone.utc)

    @staticmethod
    def sort_function_to_populate_queue(query):
        return (query.status != QueryStatus.IN_PROGESS, query.updated_at.timestamp())
