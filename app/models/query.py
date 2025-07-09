from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Generic, TypeVar

from fastapi import HTTPException

from app.config import settings
from app.models.filter import Filter
from app.models.mongo import BaseDatabaseEntity, MongoDocument
from app.schemas.enums import SupportedAssetType, AssetTypeQueryParam, QueryStatus
from app.schemas.query import (
    BaseUserQueryResponse,
    FilteredUserQueryResponse,
    OldBaseUserQueryResponse,
    OldFilteredUserQueryResponse,
    OldRecommenderUserQueryResponse,
    OldSimpleUserQueryResponse,
    RecommenderUserQueryResponse,
    ReturnedAsset,
    SimpleUserQueryResponse,
)
from app.services.helper import utc_now
from app.schemas.search_results import AssetResults


Response = TypeVar("Response", bound=BaseUserQueryResponse)
# TODO get rid of these types later on
OldResponse = TypeVar("OldResponse", bound=OldBaseUserQueryResponse)


class BaseUserQuery(MongoDocument, BaseDatabaseEntity, Generic[Response, OldResponse], ABC):
    topk: int
    status: QueryStatus = QueryStatus.QUEUED
    result_set: AssetResults | None = None
    expires_at: datetime | None = None

    def update_status(self, status: QueryStatus) -> None:
        self.status = status
        self.updated_at = utc_now()

        if self.status in (QueryStatus.COMPLETED, QueryStatus.FAILED):
            self.expires_at = (
                datetime.now(tz=timezone.utc)
                + timedelta(minutes=settings.QUERY_EXPIRATION_TIME_IN_MINUTES)
            ).replace(tzinfo=None)

    def prepare_response_kwargs(
        self, return_entire_assets: bool = False, old_schema: bool = False
    ) -> dict:
        kwargs = self.model_dump()
        if self.expires_at is not None:
            kwargs["expires_at"] = self.expires_at.replace(tzinfo=timezone.utc)

        if old_schema:
            if kwargs.get("asset_type", None) is not None:
                kwargs["asset_type"] = SupportedAssetType(kwargs["asset_type"].value)
            if kwargs.get("output_asset_type", None) is not None:
                kwargs["output_asset_type"] = SupportedAssetType(kwargs["output_asset_type"].value)

        if self.status != QueryStatus.COMPLETED:
            return kwargs
        elif self.result_set is None:
            raise ValueError("The search results are not available for this completed query")
        else:
            if old_schema:
                kwargs.update(
                    {
                        "result_asset_ids": self.result_set.asset_ids,
                        "returned_asset_count": len(self.result_set.asset_ids),
                    }
                )
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
        return self.expires_at is not None and self.expires_at < utc_now()

    @staticmethod
    def sort_function_to_populate_queue(query: BaseUserQuery) -> tuple[bool, float]:
        return (query.status != QueryStatus.IN_PROGRESS, query.updated_at.timestamp())

    @abstractmethod
    def map_to_response(
        self, return_entire_assets: bool = False, old_schema: bool = False
    ) -> Response | OldResponse:
        raise NotImplementedError


class SimpleUserQuery(BaseUserQuery[SimpleUserQueryResponse, OldSimpleUserQueryResponse]):
    search_query: str
    asset_type: AssetTypeQueryParam

    class Settings:
        name = "simpleUserQueries"

    def map_to_response(
        self, return_entire_assets: bool = False, old_schema: bool = False
    ) -> SimpleUserQueryResponse | OldSimpleUserQueryResponse:
        if old_schema:
            if self.asset_type.is_all():
                raise HTTPException(
                    status_code=400,
                    detail=f"This endpoint (v1) doesn't work with the 'ALL' asset type functionality (v2).",
                )
            return OldSimpleUserQueryResponse(
                **self.prepare_response_kwargs(return_entire_assets=False, old_schema=True)
            )
        return SimpleUserQueryResponse(
            **self.prepare_response_kwargs(return_entire_assets, old_schema=False)
        )
        

class FilteredUserQuery(BaseUserQuery[FilteredUserQueryResponse, OldFilteredUserQueryResponse]):
    search_query: str
    asset_type: SupportedAssetType
    filters: list[Filter] | None = None

    class Settings:
        name = "filteredUserQueries"

    @property
    def invoke_llm_for_parsing(self) -> bool:
        return self.filters is None

    def map_to_response(
        self, return_entire_assets: bool = False, old_schema: bool = False
    ) -> FilteredUserQueryResponse | OldFilteredUserQueryResponse:
        if old_schema:
            return OldFilteredUserQueryResponse(
                **self.prepare_response_kwargs(return_entire_assets=False, old_schema=True)
            )
        return FilteredUserQueryResponse(
            **self.prepare_response_kwargs(return_entire_assets, old_schema=False)
        )


class RecommenderUserQuery(
    BaseUserQuery[RecommenderUserQueryResponse, OldRecommenderUserQueryResponse]
):
    asset_id: str
    asset_type: SupportedAssetType
    output_asset_type: AssetTypeQueryParam

    class Settings:
        name = "recommenderUserQueries"

    def map_to_response(
        self, return_entire_assets: bool = False, old_schema: bool = False
    ) -> RecommenderUserQueryResponse | OldRecommenderUserQueryResponse:
        if old_schema:
            if self.output_asset_type.is_all():
                raise HTTPException(
                    status_code=400,
                    detail=f"This endpoint (v1) doesn't work with the 'ALL' output asset type functionality (v2).",
                )
            return OldRecommenderUserQueryResponse(
                **self.prepare_response_kwargs(return_entire_assets=False, old_schema=True)
            )
        return RecommenderUserQueryResponse(
            **self.prepare_response_kwargs(return_entire_assets, old_schema=False)
        )
