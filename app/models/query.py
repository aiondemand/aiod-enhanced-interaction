from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Generic, TypeVar

from beanie import Document

from app.config import settings
from app.models.db_entity import BaseDatabaseEntity
from app.models.filter import Filter
from app.schemas.enums import AssetType, QueryStatus
from app.schemas.query import (
    BaseUserQueryResponse,
    FilteredUserQueryResponse,
    RecommenderUserQueryResponse,
    SimpleUserQueryResponse,
)
from app.schemas.search_results import SearchResults
from app.services.helper import utc_now

Response = TypeVar("Response", bound=BaseUserQueryResponse)


class BaseUserQuery(Document, BaseDatabaseEntity, Generic[Response], ABC):
    asset_type: AssetType
    topk: int
    status: QueryStatus = QueryStatus.QUEUED
    result_set: SearchResults | None = None
    expires_at: datetime | None = None

    def update_status(self, status: QueryStatus) -> None:
        self.status = status
        self.updated_at = utc_now()

        if self.status in (QueryStatus.COMPLETED, QueryStatus.FAILED):
            self.expires_at = (
                datetime.now(tz=timezone.utc)
                + timedelta(minutes=settings.QUERY_EXPIRATION_TIME_IN_MINUTES)
            ).replace(tzinfo=None)

    def prepare_response_kwargs(self) -> dict:
        kwargs = self.model_dump()
        if self.expires_at is not None:
            kwargs["expires_at"] = self.expires_at.replace(tzinfo=timezone.utc)

        kwargs.update(self._add_results_kwargs())
        return kwargs

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

    @property
    def is_expired(self) -> bool:
        return self.expires_at is not None and self.expires_at < utc_now()

    @staticmethod
    def sort_function_to_populate_queue(query: BaseUserQuery) -> tuple[bool, float]:
        return (query.status != QueryStatus.IN_PROGRESS, query.updated_at.timestamp())

    @abstractmethod
    def map_to_response(self) -> Response:
        raise NotImplementedError


class SimpleUserQuery(BaseUserQuery[SimpleUserQueryResponse]):
    search_query: str

    class Settings:
        name = "simpleUserQueries"

    def map_to_response(self) -> SimpleUserQueryResponse:
        return SimpleUserQueryResponse(**self.prepare_response_kwargs())


class FilteredUserQuery(BaseUserQuery[FilteredUserQueryResponse]):
    search_query: str
    filters: list[Filter] | None = None

    class Settings:
        name = "filteredUserQueries"

    @property
    def invoke_llm_for_parsing(self) -> bool:
        return self.filters is None

    def map_to_response(self) -> FilteredUserQueryResponse:
        return FilteredUserQueryResponse(**self.prepare_response_kwargs())


class RecommenderUserQuery(BaseUserQuery[RecommenderUserQueryResponse]):
    asset_id: int
    output_asset_type: AssetType

    class Settings:
        name = "recommenderUserQueries"

    def map_to_response(self) -> RecommenderUserQueryResponse:
        return RecommenderUserQueryResponse(**self.prepare_response_kwargs())
