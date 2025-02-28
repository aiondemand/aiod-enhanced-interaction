from __future__ import annotations
import sys

from typing import Callable
from pymilvus import MilvusClient
from tqdm import tqdm
import json
import os
import os
from torch.utils.data import Dataset, DataLoader
from chromadb.api.client import Client as ChromaClient
import numpy as np
import random
from nltk.corpus import words

from data_types import QueryDatapoint, VectorDbClient


class AIoD_Documents(Dataset):
    def __init__(
    self, dirpath: str, include_ids: np.ndarray | None = None,
    testing_random_texts: bool = False
) -> None:
        self.dirpath = dirpath
        self.all_document_ids = np.array(
            [int(f[:f.rfind(".")]) for f in sorted(os.listdir(dirpath))]
        )

        self.split_document_ids = self.all_document_ids
        if include_ids is not None:
            mask = np.isin(include_ids, self.all_document_ids)
            self.split_document_ids = include_ids[mask]

        self.testing_random_texts = testing_random_texts

    def __getitem__(self, idx: int) -> tuple[str, int]:
        doc_id = self.split_document_ids[idx]
        with open(os.path.join(self.dirpath, f"{doc_id}.txt")) as f:
            text = f.read()

        if self.testing_random_texts:
            word_list = words.words()
            random_words = random.sample(word_list, 10_000)
            text = " ".join(random_words)

        return text, str(doc_id)

    def __len__(self) -> int:
        return len(self.split_document_ids)

    def build_loader(self, loader_kwargs: dict | None = None) -> DataLoader:
        if loader_kwargs is None:
            loader_kwargs = {
                "batch_size": 1,
                "num_workers": 1
            }
        return DataLoader(self, **loader_kwargs)

    def filter_out_already_computed_docs(
        self, client: VectorDbClient, collection_name: str
    ) -> None:
        if type(client) == ChromaClient:
            if collection_name not in client.list_collections():
                return
            collection = client.get_collection(collection_name)
            metadatas = collection.get(include=["metadatas"])["metadatas"]
            computed_doc_ids = np.unique(np.array([m["doc_id"] for m in metadatas]))
        elif type(client) == MilvusClient:
            if client.has_collection(collection_name) is False:
                return
            results = client.query(
                collection_name=collection_name,
                filter="id > -1",
                output_fields=["doc_id"]
            )
            computed_doc_ids = np.unique(np.array([doc["doc_id"] for doc in results]))
        else:
            raise ValueError("Invalid DB client")

        self.split_document_ids = self.split_document_ids[
            ~np.isin(self.split_document_ids, computed_doc_ids)
        ]


class Queries(Dataset):
    def __init__(
        self,
        json_paths: str | list[str] | None = None,
        queries: list[dict] | None = None
    ) -> None:
        if json_paths is None and queries is None or queries == []:
            raise ValueError("You need to define source of queries")
        data = queries
        if json_paths is not None:
            if type(json_paths) == str:
                json_paths = [json_paths]
            data = []
            for path in json_paths:
                with open(path) as f:
                    data.extend(json.load(f))

        self.queries = [QueryDatapoint(**d) for d in data]

    def __getitem__(self, idx: int) -> QueryDatapoint:
        return self.queries[idx]

    def __len__(self) -> int:
        return len(self.queries)

    def get_by_id(self, id: str) -> QueryDatapoint | None:
        rs = [q for q in self.queries if q.id == id]
        if rs == []:
            return None
        return rs[0]

    def build_loader(self, loader_kwargs: dict | None = None) -> DataLoader:
        if loader_kwargs is None:
            loader_kwargs = {
                "batch_size": 1,
                "num_workers": 1
            }
        return DataLoader(self, collate_fn=lambda x: x, **loader_kwargs)


def process_documents_and_store_to_filesystem(
    client: VectorDbClient, collection_name: str,
    extraction_function: Callable[[dict], str],
    savedir: str, docs_window_size: int = 10_000,
    extension: str = ".txt"
) -> None:
    os.makedirs(savedir, exist_ok=True)

    if type(client) == ChromaClient:
        collection = client.get_collection(collection_name)

        for offset in tqdm(range(0, collection.count(), docs_window_size)):
            docs = collection.get(
                limit=docs_window_size, offset=offset, include=["metadatas"]
            )
            doc_ids, metadata = docs["ids"], docs["metadatas"]

            objects = [json.loads(d["json_string"]) for d in metadata]
            extracted_texts = [extraction_function(o) for o in objects]

            for id, text in zip(doc_ids, extracted_texts):
                with open(os.path.join(savedir, f"{id}{extension}"), "w") as f:
                    f.write(text)

    elif type(client) == MilvusClient:
        raise ValueError(
            "We dont support this function utilizing Milvus database as there's apparently a hard cap of 2^16 characters for strings." +
            "Due to this limitation, we are unable to store the stringified JSONs of the assets in the vector database (which is not requested in the first place I assume)"
        )
    else:
        raise ValueError("Invalid DB client")
