import json
import pandas as pd
import os
import sys
from torch.utils.data import DataLoader
from chromadb.api.client import Client
from chromadb import Collection
from tqdm import tqdm
import torch
import uuid
import numpy as np

from model.lm_encoders.setup import ModelSetup
from model.base import EmbeddingModel
import utils
from dataset import AIoD_Documents

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(src_dir)


class Chroma_EmbeddingStore:
    def __init__(self, client: Client, verbose: bool = False) -> None:
        self.client = client
        self.verbose = verbose

    def _get_collection(
        self, collection_name: str, create_collection: bool = False
    ) -> Collection:
        try:
            collection = self.client.get_collection(collection_name)
        except Exception as e:
            if create_collection is False:
                print(f"Collection '{collection_name}' doesn't exist.")
                raise e
            collection = self.create_collection(collection_name)
        
        return collection

    def create_collection(self, collection_name: str) -> Collection:
        return self.client.create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine"
            },
            get_or_create=True
        )

    def store_embeddings(
        self, model: EmbeddingModel, loader: DataLoader, collection_name: str,
        chroma_batch_size: int = 50
    ) -> None:
        was_training = model.training
        model.eval()
        collection = self._get_collection(collection_name, create_collection=True)

        all_embeddings = []
        all_ids = []
        all_meta = []
        for it, (texts, doc_ids) in tqdm(
            enumerate(loader), total=len(loader), disable=self.verbose is False
        ):
            with torch.no_grad():
                embeddings = model(texts)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_ids.extend([str(uuid.uuid4()) for _ in range(len(doc_ids))])
            all_meta.extend([{"doc_id": id} for id in doc_ids])
    
            if len(all_embeddings) == chroma_batch_size or it == len(loader) - 1:
                all_embeddings = np.vstack(all_embeddings)
                collection.add(
                    embeddings=all_embeddings, 
                    ids=all_ids,
                    metadatas=all_meta
                )

                all_embeddings = []
                all_ids = []
                all_meta = []

        if was_training:
            model.train()

    def semantic_search(
        self, model: EmbeddingModel, query_list: list[tuple[str, int]],
        emb_collection_name: str, topk: int = 10, chroma_batch_size: int = 50
    ) -> list:
        was_training = model.training
        model.eval()
        collection = self._get_collection(emb_collection_name)

        all_results = {
            "doc_ids": [],
            "distances": [],

        }
        all_embeddings = []        
        query_loader = DataLoader(
            query_list, collate_fn=AIoD_Documents._collate_fn, batch_size=4, num_workers=2
        )
        for it, (texts, _) in tqdm(
            enumerate(query_loader), total=len(query_loader), disable=self.verbose is False
        ):
            with torch.no_grad():
                embeddings = model(texts)
            all_embeddings.append(embeddings.cpu().numpy())
            
            if len(all_embeddings) == chroma_batch_size or it == len(query_loader) - 1:
                all_embeddings = np.vstack(all_embeddings)

                sem_search_results = collection.query(
                    query_embeddings=all_embeddings,
                    n_results=topk,
                    include=["metadatas", "distances"]
                )
                doc_ids = [
                    [
                        doc["doc_id"] for doc in q_docs
                    ] 
                    for q_docs in sem_search_results["metadatas"]
                ]

                all_results["doc_ids"].extend(doc_ids)
                all_results["distances"].extend(sem_search_results["distances"])

                all_embeddings = []
        
        if was_training:
            model.train()
        return all_results
    
    def retrieve_documents_from_result_set(
        self, result_set: dict[str, list[list[str] | list[int]]],
        document_collection_name: str
    ) -> None:
        all_docs = []
        col = self.client.get_collection(document_collection_name)

        for doc_ids in result_set["doc_ids"]:
            revert_indices = np.argsort(
                pd.Series(doc_ids).sort_values().index
            )
            response = col.get(result_set["doc_ids"][0])["metadatas"]
            documents = [
                json.loads(meta["json_string"])
                for meta in np.array(response)[revert_indices]
            ]
            all_docs.append(documents)

        return all_docs


def compute_embeddings_wrapper(
    client: Client, model: EmbeddingModel, text_dirpath: str, new_collection_name: str
) -> None:
    ds = AIoD_Documents(text_dirpath)
    loader = ds.build_loader({"batch_size": 16, "num_workers": 2})

    store = Chroma_EmbeddingStore(client, verbose=True)
    store.store_embeddings(model, loader, new_collection_name)
    

if __name__ == "__main__":
    client = utils.init()
    model = ModelSetup._setup_sentence_transformer_hierarchical(
        model_path="BAAI/bge-base-en-v1.5",
        max_num_chunks=5,
        use_chunk_transformer=False,
        pooling="mean", 
        parallel_chunk_processing=True
    )
    text_dirpath = "data/texts"
    collection_name = "embeddings-BAAI-simple"

    compute_embeddings_wrapper(client, model, text_dirpath, collection_name)

    # Perform semantic search
    # store = Chroma_EmbeddingStore(client, verbose=True)
    # QUERY = "I want a dataset about movies reviews"
    # query_list = [(QUERY, 0)]
    # results = store.semantic_search(model, query_list, collection_name)
    # top_docs = store.retrieve_documents_from_result_set(
    #     results, document_collection_name="datasets"
    # )