from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field
from functools import reduce

from app.schemas.enums import SupportedAssetType


class SearchResults(BaseModel):
    asset_ids: list[int] = Field(default_factory=list)
    distances: list[float] = Field(default_factory=list)
    asset_types: list[SupportedAssetType] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.asset_ids)


class AssetResults(SearchResults):
    assets: list[dict] = Field(default_factory=list)

    def __add__(self, other: AssetResults) -> AssetResults:
        if isinstance(other, AssetResults) is False:
            raise TypeError("Invalid object type")

        sum_results = AssetResults(
            asset_ids=self.asset_ids + other.asset_ids,
            distances=self.distances + other.distances,
            asset_types=self.asset_types + other.asset_types,
            assets=self.assets + other.assets,
        )
        return sum_results[np.argsort(sum_results.distances).tolist()]

    def __getitem__(self, idx: int | slice | list[int]) -> AssetResults:
        if isinstance(idx, (int, slice)):
            if isinstance(idx, int):
                idx = slice(idx, idx + 1)

            return AssetResults(
                asset_ids=self.asset_ids[idx],
                distances=self.distances[idx],
                asset_types=self.asset_types[idx],
                assets=self.assets[idx],
            )
        elif isinstance(idx, list):
            return AssetResults(
                asset_ids=[self.asset_ids[i] for i in idx],
                distances=[self.distances[i] for i in idx],
                asset_types=[self.asset_types[i] for i in idx],
                assets=[self.assets[i] for i in idx],
            )
        else:
            raise TypeError("Invalid index type")

    def filter_out_assets_by_id(self, asset_ids_to_del: list[int]) -> AssetResults:
        idx_to_keep = np.where(~np.isin(self.asset_ids, asset_ids_to_del))[0].tolist()
        return self[idx_to_keep]

    @staticmethod
    def merge_results(asset_results: list[AssetResults], k: int) -> AssetResults:
        all_results = reduce(lambda a, b: a + b, asset_results)
        return all_results[:k]
