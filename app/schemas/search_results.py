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

        all_ids = self.asset_ids + other.asset_ids
        all_distances = self.distances + other.distances
        all_asset_types = self.asset_types + other.asset_types
        all_assets = self.assets + other.assets

        new_idx_order = np.argsort(all_distances)

        return AssetResults(
            asset_ids=[all_ids[idx] for idx in new_idx_order],
            distances=[all_distances[idx] for idx in new_idx_order],
            asset_types=[all_asset_types[idx] for idx in new_idx_order],
            assets=[all_assets[idx] for idx in new_idx_order],
        )

    def __getitem__(self, idx: int) -> dict:
        return {
            "asset_ids": self.asset_ids[idx],
            "distances": self.distances[idx],
            "asset_types": self.asset_types[idx],
            "assets": self.assets[idx],
        }

    def __len__(self) -> int:
        return len(self.asset_ids)

    def filter_out_assets_by_id(self, asset_ids_to_del: list[int]) -> AssetResults:
        idx_to_keep = np.where(~np.isin(self.asset_ids, asset_ids_to_del))[0]

        return AssetResults(
            asset_ids=[self.asset_ids[idx] for idx in idx_to_keep],
            distances=[self.distances[idx] for idx in idx_to_keep],
            asset_types=[self.asset_types[idx] for idx in idx_to_keep],
            assets=[self.assets[idx] for idx in idx_to_keep],
        )

    @staticmethod
    def merge_results(asset_results: list[AssetResults], k: int) -> AssetResults:
        all_results = reduce(lambda a, b: a + b, asset_results)
        topk_results = [all_results[idx] for idx in range(k)]

        kwargs: dict[str, list] = reduce(
            lambda acc, item: {key: acc.get(key, []) + [item[key]] for key in item},
            topk_results,
            {},
        )
        return AssetResults(**kwargs)
