from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field
from functools import reduce

from app.schemas.enums import SupportedAssetType


class SearchResults(BaseModel):
    asset_ids: list[int] = Field(default_factory=list)
    distances: list[float] = Field(default_factory=list)
    asset_types: list[SupportedAssetType] = Field(default_factory=list)

    def __add__(self, other: SearchResults) -> SearchResults:
        if isinstance(other, SearchResults) is False:
            raise TypeError("Invalid object type")

        all_ids = self.asset_ids + other.asset_ids
        all_distances = self.distances + other.distances
        all_asset_types = self.asset_types + other.asset_types
        new_idx_order = np.argsort(all_distances)

        return SearchResults(
            asset_ids=[all_ids[idx] for idx in new_idx_order],
            distances=[all_distances[idx] for idx in new_idx_order],
            asset_types=[all_asset_types[idx] for idx in new_idx_order],
        )

    def __getitem__(self, idx: int) -> dict:
        return {
            "asset_ids": self.asset_ids[idx],
            "distances": self.distances[idx],
            "asset_types": self.asset_types[idx],
        }

    def __len__(self) -> int:
        return len(self.asset_ids)

    def filter_out_assets_by_id(self, asset_ids_to_del: list[int]) -> SearchResults:
        idx_to_keep = np.where(~np.isin(self.asset_ids, asset_ids_to_del))[0]

        return SearchResults(
            asset_ids=[self.asset_ids[idx] for idx in idx_to_keep],
            distances=[self.distances[idx] for idx in idx_to_keep],
            asset_types=[self.asset_types[idx] for idx in idx_to_keep],
        )

    @staticmethod
    def merge_results(search_results: list[SearchResults], k: int) -> SearchResults:
        all_results = reduce(lambda a, b: a + b, search_results)
        topk_results = [all_results[idx] for idx in range(k)]

        kwargs: dict[str, list] = reduce(
            lambda acc, item: {key: acc.get(key, []) + [item[key]] for key in item},
            topk_results,
            {},
        )
        return SearchResults(**kwargs)
