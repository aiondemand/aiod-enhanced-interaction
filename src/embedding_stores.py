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
from abc import ABC, abstractmethod 
from sentence_transformers.util import semantic_search
from sklearn.metrics import ndcg_score

from model.lm_encoders.setup import ModelSetup
from model.base import EmbeddingModel
import utils
from dataset import AIoD_Documents

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(src_dir)


class EmbeddingStore(ABC):
    @abstractmethod
    def store_embeddings(
        self, model: EmbeddingModel, loader: DataLoader, **kwargs
    ) -> None:
        pass

    @abstractmethod
    def retrieve_topk_documents(
        self, model: EmbeddingModel, query_loader: DataLoader, topk: int = 10, **kwargs
    ) -> dict[str, list[list[str] | list[float]]]:
        pass
        

class Chroma_EmbeddingStore(EmbeddingStore):
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

    def retrieve_topk_documents(
        self, model: EmbeddingModel, query_loader: DataLoader, topk: int = 10,
        emb_collection_name: str | None = None, chroma_batch_size: int = 50
    ) -> dict[str, list[list[str] | list[float]]]:
        was_training = model.training
        model.eval()
        collection = self._get_collection(emb_collection_name)

        all_results = {
            "doc_ids": [],
            "distances": [],

        }
        all_embeddings = []        
        
        for it, (texts, _) in tqdm(
            enumerate(query_loader), 
            total=len(query_loader), 
            disable=self.verbose is False
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
    
    def _retrieve_documents_from_result_set(
        self, result_set: dict[str, list[list[str] | list[int]]],
        document_collection_name: str
    ) -> dict:
        all_docs = []
        col = self.client.get_collection(document_collection_name)

        for doc_ids in result_set["doc_ids"]:
            revert_indices = np.argsort(
                pd.Series(doc_ids).sort_values().index
            )
            response = col.get(doc_ids)["metadatas"]
            documents = [
                json.loads(meta["json_string"])
                for meta in np.array(response)[revert_indices]
            ]
            all_docs.append(documents)

        return all_docs
    

class Filesystem_EmbeddingStore(EmbeddingStore):
    def __init__(self, save_dirpath: str) -> None:
        self.save_dirpath = save_dirpath
        self.vector_store_in_memory = None
    
    def store_embeddings(
        self, model: EmbeddingModel, loader: DataLoader
    ) -> None:
        was_training = model.training
        model.eval()
        os.makedirs(self.save_dirpath, exist_ok=True)

        for texts, doc_ids in tqdm(loader):
            with torch.no_grad():
                embeddings = model(texts)

            for id, emb in zip(doc_ids, embeddings):
                filepath = os.path.join(self.save_dirpath, f"{id}.pt")
                torch.save(emb, filepath)

        if was_training:
            model.train()

    # TODO change
    def retrieve_topk_documents(
        self, model: EmbeddingModel, query_loader: DataLoader, topk: int = 10
    ) -> dict[str, list[list[str] | list[float]]]:
        if self.vector_store_in_memory is None:
            self.vector_store_in_memory = self._load_embeddings()        
        all_document_ids = np.array([
            file[:file.rfind(".")]
            for file in sorted(os.listdir(self.save_dirpath))
        ])

        all_results_sets = []
        for texts, _ in tqdm(query_loader):
            with torch.no_grad():
                query_emb = model(texts)
    
            all_results_sets.extend(semantic_search(
                query_emb, self.vector_store_in_memory, query_chunk_size=100, 
                corpus_chunk_size=10_000, top_k=topk
            ))
    
        topk_docs_ids = []
        topk_distances = []
        for db_matches in all_results_sets:
            db_indices = [db_match["corpus_id"] for db_match in db_matches]
            db_scores = [db_match["score"] for db_match in db_matches]

            topk_docs_ids.append(all_document_ids[db_indices].tolist())
            topk_distances.append((1 - np.array(db_scores)).tolist())

        results = {
            "doc_ids": topk_docs_ids,
            "distances": topk_distances
        }
        return results

    def _load_embeddings(self) -> torch.Tensor:
        if (
            os.path.exists(self.save_dirpath) is False 
            or len(os.listdir(self.save_dirpath)) == 0
        ):
            return None

        all_embeddings = []
        for filename in sorted(os.listdir(self.save_dirpath)):
            emb = torch.load(
                os.path.join(self.save_dirpath, filename), 
                utils.get_device()
            )
            all_embeddings.append(emb)

        return torch.vstack(all_embeddings)


def compute_embeddings_wrapper(
    client: Client, model: EmbeddingModel, 
    text_dirpath: str, new_collection_name: str,
    loader_kwargs: dict | None = None
) -> None:
    ds = AIoD_Documents(text_dirpath, testing_random_texts=False)
    ds.filter_out_already_computed_docs(client, new_collection_name)
    loader = ds.build_loader(loader_kwargs)

    store = Chroma_EmbeddingStore(client, verbose=True)
    store.store_embeddings(model, loader, new_collection_name, chroma_batch_size=50)
    
if __name__ == "__main__":
    client = utils.init()

    # 6h20m
    # model = ModelSetup._setup_gte_large(model_max_length=4096) # batch: 8 (non-hierarchical)
    
    text_dirpath = "data/texts"
    collection_name = "embeddings-gte_large-simple-v0"

    # compute_embeddings_wrapper(
    #     client, model, text_dirpath, collection_name, 
    #     loader_kwargs={
    #         "batch_size": batch_size,
    #         "num_workers": 2
    #     }
    # )

    #######################

    # 6h40m
    # batch_size = 32
    # model = ModelSetup._setup_multilingual_e5_large() # batch: 32 (hierarchical)

    # text_dirpath = "data/texts"
    # collection_name = "embeddings-multilingual_e5_large-simple-v0"

    # compute_embeddings_wrapper(
    #     client, model, text_dirpath, collection_name, 
    #     loader_kwargs={
    #         "batch_size": batch_size,
    #         "num_workers": 2
    #     }
    # )

    # Perform semantic search
    store = Chroma_EmbeddingStore(client, verbose=True)
    
    model = ModelSetup.setup_hierarchical_model(
        model_path="BAAI/bge-base-en-v1.5",
        max_num_chunks=5,
        use_chunk_transformer=False,
        token_pooling="none",
        chunk_pooling="mean", 
        parallel_chunk_processing=True
    )
        
    collection_name = "embeddings-BAAI-simple"

    QUERY = "I want a dataset about movies reviews"
    query_list = [(QUERY, 0)]
    query_loader = DataLoader(
        query_list, collate_fn=AIoD_Documents._collate_fn, 
        batch_size=4, num_workers=2
    )

    results = store.retrieve_topk_documents(
        model, query_loader, topk=10, emb_collection_name=collection_name
    )
    results = {
        "doc_ids": [["strings"]], #2D
        "distances": [["floats"]] #2D
    }

    top_docs = store._retrieve_documents_from_result_set(
        results, document_collection_name="datasets"
    )