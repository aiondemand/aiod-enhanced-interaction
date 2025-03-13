from __future__ import annotations

from datetime import datetime

from app.schemas.enums import AssetType
from pydantic import BaseModel, Field


class RequestParams(BaseModel):
    offset: int = 0
    limit: int
    from_time: datetime | None = None
    to_time: datetime | None = None

    def new_page(self, offset: int | None, limit: int | None) -> RequestParams:
        new_obj = RequestParams(**self.model_dump())

        if offset is not None:
            new_obj.offset = offset
        if limit is not None:
            new_obj.limit = limit
        return new_obj


class DbSearchParams(BaseModel):
    # TODO create an abstract class
    pass


class MilvusSearchParams(BaseModel):
    data: list[list[float]]
    topk: int
    group_by_field: str = "doc_id"
    output_fields: list[str] = Field(default_factory=lambda: ["doc_id"])
    search_params: dict = Field(default_factory=lambda: {"metric_type": "COSINE"})
    asset_type: AssetType

    # filter related attributes
    metadata_filter: str = ""
    doc_ids_to_exclude: list[str] = Field(default_factory=list)

    @property
    def filter(self) -> str:
        if len(self.doc_ids_to_exclude) > 0 and len(self.metadata_filter) > 0:
            return f"({self.metadata_filter}) and (doc_id not in {self.doc_ids_to_exclude})"
        elif len(self.doc_ids_to_exclude) > 0:
            return f"doc_id not in {self.doc_ids_to_exclude}"
        elif len(self.metadata_filter) > 0:
            return self.metadata_filter
        else:
            return ""

    def get_params(self) -> dict:
        return {
            "data": self.data,
            "topk": self.topk,
            "group_by_field": self.group_by_field,
            "output_fields": self.output_fields,
            "search_params": self.search_params,
            "filter": self.filter,
        }
