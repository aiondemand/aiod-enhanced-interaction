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

import utils
from preprocess.text_operations import ConvertJsonToString


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
    
    def build_loader(self, loader_kwargs: dict | None) -> DataLoader:
        if loader_kwargs is None:
            loader_kwargs = {
                "batch_size": 1,
                "num_workers": 1
            }
        return DataLoader(self, collate_fn=self._collate_fn, **loader_kwargs)
    
    @staticmethod
    def _collate_fn(
        batch: list[tuple[str, int]]
    ) -> tuple[list[str], list[str]]:
        all_texts = [x[0] for x in batch]
        all_ids = [x[1] for x in batch]
        
        return all_texts, all_ids


def process_documents_and_store_to_filesystem(
    client: Client, collection_name: str, 
    extraction_function: Callable[[dict], str],
    savedir: str, docs_window_size: int = 10_000
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
            with open(os.path.join(savedir, f"{id}.txt"), "w") as f:
                f.write(text)


if __name__ == "__main__":
    client = utils.init(return_chroma_client=True)
    savedir = "./data/extracted_data"
    collection_name = "datasets"

    process_documents_and_store_to_filesystem(
        client, collection_name, savedir=savedir,
        extraction_function=ConvertJsonToString.extract_relevant_info,
    )