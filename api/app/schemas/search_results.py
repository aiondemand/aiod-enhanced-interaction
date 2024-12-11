from __future__ import annotations

import numpy as np
from pydantic import BaseModel


class SearchResults(BaseModel):
    doc_ids: list[str] = []
    distances: list[float] = []

    def __add__(self, other: SearchResults) -> SearchResults:
        if isinstance(other, SearchResults) is False:
            raise TypeError("Invalid object type")

        other = other.filter_out_docs(self.doc_ids)

        all_ids = self.doc_ids + other.doc_ids
        all_distances = self.distances + other.distances
        new_idx_order = np.argsort(all_distances)

        return SearchResults(
            doc_ids=[all_ids[idx] for idx in new_idx_order],
            distances=[all_distances[idx] for idx in new_idx_order],
        )

    def __len__(self) -> int:
        return len(self.doc_ids)

    def filter_out_docs(self, doc_ids_to_del: list[str]) -> SearchResults:
        idx_to_keep = np.where(~np.isin(self.doc_ids, doc_ids_to_del))[0]

        return SearchResults(
            doc_ids=[self.doc_ids[idx] for idx in idx_to_keep],
            distances=[self.distances[idx] for idx in idx_to_keep],
        )
