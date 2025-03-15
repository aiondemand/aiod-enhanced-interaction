from __future__ import annotations

from abc import abstractmethod
from datetime import datetime, timezone
from functools import partial
from typing import Type
from uuid import uuid4

from pydantic import BaseModel, Field

from app.models.filter import Filter
from app.schemas.enums import AssetType, QueryStatus
from app.schemas.query import (
    BaseUserQueryResponse,
    FilteredUserQueryResponse,
    RecommenderUserQueryResponse,
    SimpleUserQueryResponse,
)
from app.schemas.search_results import SearchResults


class BaseUserQuery(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))

    asset_type: AssetType
    topk: int
    status: QueryStatus = QueryStatus.QUEUED

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

        asset_ids = self.result_set.asset_ids
        return response_model(
            returned_asset_count=len(asset_ids),
            result_asset_ids=asset_ids,
            **self.model_dump(),
        )

    @staticmethod
    def sort_function_to_populate_queue(query: BaseUserQuery) -> tuple[bool, float]:
        return (query.status != QueryStatus.IN_PROGRESS, query.updated_at.timestamp())

    @abstractmethod
    def map_to_response(self) -> BaseUserQueryResponse:
        raise NotImplementedError


class SimpleUserQuery(BaseUserQuery):
    search_query: str

    def map_to_response(self) -> SimpleUserQueryResponse:
        return self._map_to_response(SimpleUserQueryResponse)


class FilteredUserQuery(BaseUserQuery):
    search_query: str
    filters: list[Filter] | None = None

    @property
    def invoke_llm_for_parsing(self) -> bool:
        return self.filters is None

    def map_to_response(self) -> FilteredUserQueryResponse:
        return self._map_to_response(FilteredUserQueryResponse)


class RecommenderUserQuery(BaseUserQuery):
    asset_id: int
    output_asset_type: AssetType

    def map_to_response(self) -> RecommenderUserQueryResponse:
        return self._map_to_response(RecommenderUserQueryResponse)
