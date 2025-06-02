from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Generic, TypeVar

from app.config import settings
from app.models.db_entity import DatabaseEntity
from app.models.filter import Filter
from app.schemas.enums import SupportedAssetType, AssetTypeQueryParam, QueryStatus
from app.schemas.query import (
    BaseUserQueryResponse,
    FilteredUserQueryResponse,
    RecommenderUserQueryResponse,
    ReturnedAsset,
    SimpleUserQueryResponse,
)
from app.schemas.search_results import AssetResults

Response = TypeVar("Response", bound=BaseUserQueryResponse)


class BaseUserQuery(DatabaseEntity, Generic[Response], ABC):
    topk: int
    status: QueryStatus = QueryStatus.QUEUED
    result_set: AssetResults | None = None
    expires_at: datetime | None = None

    def update_status(self, status: QueryStatus) -> None:
        self.status = status
        self.updated_at = datetime.now(tz=timezone.utc)

        if self.status in (QueryStatus.COMPLETED, QueryStatus.FAILED):
            self.expires_at = datetime.now(tz=timezone.utc) + timedelta(
                minutes=settings.QUERY_EXPIRATION_TIME_IN_MINUTES
            )

    def prepare_response_kwargs(self, return_entire_assets: bool = False) -> dict:
        kwargs = self.model_dump()

        if self.status != QueryStatus.COMPLETED:
            return kwargs
        elif self.result_set is None:
            raise ValueError("The search results are not available for this completed query")
        else:
            kwargs.update(
                {
                    "results": ReturnedAsset.create_list_from_asset_results(
                        self.result_set, return_entire_assets
                    )
                }
            )
            return kwargs

    @property
    def is_expired(self) -> bool:
        return self.expires_at is not None and self.expires_at < datetime.now(tz=timezone.utc)

    @staticmethod
    def sort_function_to_populate_queue(query: BaseUserQuery) -> tuple[bool, float]:
        return (query.status != QueryStatus.IN_PROGRESS, query.updated_at.timestamp())

    @abstractmethod
    def map_to_response(self, return_entire_assets: bool = False) -> Response:
        raise NotImplementedError


class SimpleUserQuery(BaseUserQuery[SimpleUserQueryResponse]):
    search_query: str
    asset_type: AssetTypeQueryParam

    def map_to_response(self, return_entire_assets: bool = False) -> SimpleUserQueryResponse:
        return SimpleUserQueryResponse(**self.prepare_response_kwargs(return_entire_assets))


class FilteredUserQuery(BaseUserQuery[FilteredUserQueryResponse]):
    search_query: str
    asset_type: SupportedAssetType
    filters: list[Filter] | None = None

    @property
    def invoke_llm_for_parsing(self) -> bool:
        return self.filters is None

    def map_to_response(self, return_entire_assets: bool = False) -> FilteredUserQueryResponse:
        return FilteredUserQueryResponse(**self.prepare_response_kwargs(return_entire_assets))


class RecommenderUserQuery(BaseUserQuery[RecommenderUserQueryResponse]):
    asset_id: int
    asset_type: SupportedAssetType
    output_asset_type: AssetTypeQueryParam

    def map_to_response(self, return_entire_assets: bool = False) -> RecommenderUserQueryResponse:
        return RecommenderUserQueryResponse(**self.prepare_response_kwargs(return_entire_assets))
