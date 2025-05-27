from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field


class SearchResults(BaseModel):
    asset_ids: list[int] = Field(default_factory=list)
    distances: list[float] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.asset_ids)


class AssetResults(SearchResults):
    assets: list[dict] = Field(default_factory=list)

    def __add__(self, other: AssetResults) -> AssetResults:
        if isinstance(other, AssetResults) is False:
            raise TypeError("Invalid object type")

        other = other.filter_out_assets_by_id(self.asset_ids)

        all_ids = self.asset_ids + other.asset_ids
        all_distances = self.distances + other.distances
        all_assets = self.assets + other.assets

        new_idx_order = np.argsort(all_distances)

        return AssetResults(
            asset_ids=[all_ids[idx] for idx in new_idx_order],
            distances=[all_distances[idx] for idx in new_idx_order],
            assets=[all_assets[idx] for idx in new_idx_order],
        )

    def filter_out_assets_by_id(self, asset_ids_to_del: list[int]) -> AssetResults:
        idx_to_keep = np.where(~np.isin(self.asset_ids, asset_ids_to_del))[0]

        return AssetResults(
            asset_ids=[self.asset_ids[idx] for idx in idx_to_keep],
            distances=[self.distances[idx] for idx in idx_to_keep],
            assets=[self.assets[idx] for idx in idx_to_keep],
        )
