from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field


class SearchResults(BaseModel):
    asset_ids: list[int] = Field(default_factory=list)
    distances: list[float] = Field(default_factory=list)

    def __add__(self, other: SearchResults) -> SearchResults:
        if isinstance(other, SearchResults) is False:
            raise TypeError("Invalid object type")

        other = other.filter_out_assets_by_id(self.asset_ids)

        all_ids = self.asset_ids + other.asset_ids
        all_distances = self.distances + other.distances
        new_idx_order = np.argsort(all_distances)

        return SearchResults(
            asset_ids=[all_ids[idx] for idx in new_idx_order],
            distances=[all_distances[idx] for idx in new_idx_order],
        )

    def __len__(self) -> int:
        return len(self.asset_ids)

    def filter_out_assets_by_id(self, asset_ids_to_del: list[int]) -> SearchResults:
        idx_to_keep = np.where(~np.isin(self.asset_ids, asset_ids_to_del))[0]

        return SearchResults(
            asset_ids=[self.asset_ids[idx] for idx in idx_to_keep],
            distances=[self.distances[idx] for idx in idx_to_keep],
        )
