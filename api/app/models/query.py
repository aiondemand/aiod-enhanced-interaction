from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Generic, Type, TypeVar

from app.config import settings
from app.models.db_entity import DatabaseEntity
from app.models.filter import Filter
from app.schemas.enums import AssetType, QueryStatus
from app.schemas.query import (
    BaseUserQueryResponse,
    FilteredUserQueryResponse,
    RecommenderUserQueryResponse,
    SimpleUserQueryResponse,
)
from app.schemas.search_results import SearchResults

Response = TypeVar("Response", bound=BaseUserQueryResponse)


class BaseUserQuery(DatabaseEntity, Generic[Response], ABC):
    asset_type: AssetType
    topk: int
    status: QueryStatus = QueryStatus.QUEUED
    result_set: SearchResults | None = None
    expires_at: datetime | None = None

    def update_status(self, status: QueryStatus) -> None:
        self.status = status
        self.updated_at = datetime.now(tz=timezone.utc)

        if self.status in (QueryStatus.COMPLETED, QueryStatus.FAILED):
            # an arbitrarily chosen expiration date
            self.expires_at = datetime.now(tz=timezone.utc) + timedelta(
                minutes=settings.QUERY_EXPIRATION_TIME_IN_MINUTES
            )

    def _add_results_kwargs(self) -> dict:
        if self.status != QueryStatus.COMPLETED:
            return {}
        elif self.result_set is not None:
            return {
                "returned_asset_count": len(self.result_set),
                "result_asset_ids": self.result_set.asset_ids,
            }
        else:
            raise ValueError("SearchResults are not available for this completed query")

    @staticmethod
    def sort_function_to_populate_queue(query: BaseUserQuery) -> tuple[bool, float]:
        return (query.status != QueryStatus.IN_PROGRESS, query.updated_at.timestamp())

    @abstractmethod
    def map_to_response(self) -> Response:
        raise NotImplementedError


class SimpleUserQuery(BaseUserQuery[SimpleUserQueryResponse]):
    search_query: str

    def map_to_response(self) -> SimpleUserQueryResponse:
        return SimpleUserQueryResponse(**self.model_dump(), **self._add_results_kwargs())


class FilteredUserQuery(BaseUserQuery[FilteredUserQueryResponse]):
    search_query: str
    filters: list[Filter] | None = None

    @property
    def invoke_llm_for_parsing(self) -> bool:
        return self.filters is None

    def map_to_response(self) -> FilteredUserQueryResponse:
        return FilteredUserQueryResponse(**self.model_dump(), **self._add_results_kwargs())


class RecommenderUserQuery(BaseUserQuery[RecommenderUserQueryResponse]):
    asset_id: int
    output_asset_type: AssetType

    def map_to_response(self) -> RecommenderUserQueryResponse:
        return RecommenderUserQueryResponse(**self.model_dump(), **self._add_results_kwargs())
