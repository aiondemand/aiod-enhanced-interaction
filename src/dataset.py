from __future__ import annotations

from typing import Callable
from tqdm import tqdm
import json
import os
import os
from torch.utils.data import Dataset, DataLoader
from chromadb.api.client import Client
import numpy as np
import random
from nltk.corpus import words
from pydantic import BaseModel


class AnnotatedDoc(BaseModel):
    id: str
    score: int


class QueryDatapoint(BaseModel):
    text: str
    id: str | None = None
    annotated_docs: list[AnnotatedDoc] | None = None

    def get_relevant_documents(
        self, relevance_func: Callable[[float], bool]
    ) -> list[AnnotatedDoc]:
        return [
            doc for doc in self.annotated_docs
            if relevance_func(doc.score)
        ]
    

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
        self, client: Client, collection_name: str
    ) -> None:
        try:
            collection = client.get_collection(collection_name)
            metadatas = collection.get(include=["metadatas"])["metadatas"]
            computed_doc_ids = np.array([m["doc_id"] for m in metadatas])

            self.split_document_ids = self.split_document_ids[
                ~np.isin(self.split_document_ids, computed_doc_ids)
            ]
        except:
            return
        

class Queries(Dataset):
    def __init__(
        self, json_path: str | None = None, queries: list[dict] | None = None
    ) -> None:
        if json_path is None and queries is None or queries == []:
            raise ValueError("You need to define source of queries")
        data = queries
        if json_path is not None:
            with open(json_path) as f:
                data = json.load(f)
        
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
    client: Client, collection_name: str, 
    extraction_function: Callable[[dict], str],
    savedir: str, docs_window_size: int = 10_000,
    extension: str = ".txt"
) -> None:
    os.makedirs(savedir, exist_ok=True)
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
    