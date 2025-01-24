from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel


class SearchResults(BaseModel):
    # Number of documents that match the query
    # For the simple search, this is the total number of documents in the collection
    # For the filtered search, this is the number of documents that match all the filters applied
    num_hits: int = -1

    # TODO either make the default None, or Field(default_factory=list)
    doc_ids: list[str] = []
    distances: list[float] = []
    documents: list[dict | None] = []

    def __add__(self, other: SearchResults) -> SearchResults:
        if isinstance(other, SearchResults) is False:
            raise TypeError("Invalid object type")

        all_ids = self.doc_ids + other.doc_ids
        all_distances = self.distances + other.distances
        all_documents = self.documents + other.documents

        data = pd.DataFrame(data=[all_ids, all_distances, all_documents]).T
        data.columns = ["doc_ids", "distances", "documents"]
        data = (
            data.drop_duplicates(subset=["doc_ids"])
            .dropna(subset=["documents"])
            .sort_values(by="distances")
        )

        return SearchResults(
            doc_ids=data["doc_ids"].tolist(),
            distances=data["distances"].tolist(),
            documents=data["documents"].tolist(),
        )

    def __len__(self) -> int:
        return len(self.doc_ids)

    def filter_out_docs(self) -> list[str]:
        exists_mask = np.array([doc is not None for doc in self.documents])
        doc_ids_to_del = [self.doc_ids[idx] for idx in np.where(~exists_mask)[0]]
        idx_to_keep = np.where(exists_mask)[0]

        self.doc_ids = [self.doc_ids[idx] for idx in idx_to_keep]
        self.distances = [self.distances[idx] for idx in idx_to_keep]
        self.documents = [self.documents[idx] for idx in idx_to_keep]

        return doc_ids_to_del
