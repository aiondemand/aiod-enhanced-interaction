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
from pydantic import BaseModel

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(src_dir)

from model.lm_encoders.setup import ModelSetup
from model import EmbeddingModel
import utils
from dataset import AIoD_Documents, Queries


class SemanticSearchResult(BaseModel):
    query_id: str
    doc_ids: list[str]
    distances: list[float]
    

class EmbeddingStore(ABC):
    @abstractmethod
    def store_embeddings(
        self, model: EmbeddingModel, loader: DataLoader, **kwargs
    ) -> None:
        pass

    @abstractmethod
    def retrieve_topk_documents(
        self, model: EmbeddingModel, query_loader: DataLoader, topk: int = 10, 
        save_dirpath: str | None = None, load_dirpath: str | None = None, **kwargs
    ) -> list[SemanticSearchResult]:
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
        save_dirpath: str | None = None, load_dirpath: str | None = None,
        emb_collection_name: str | None = None, chroma_batch_size: int = 50,
    ) -> list[SemanticSearchResult]:
        if load_dirpath is not None:
            try:
                topk_store = LocalTopKDocumentsStore(load_dirpath, topk=topk)
                return topk_store.load_topk_documents(query_loader)
            except:
                pass

        was_training = model.training
        model.eval()
        collection = self._get_collection(emb_collection_name)

        all_results = []
        all_embeddings = []
        all_query_ids = []
        
        for it, queries in tqdm(
            enumerate(query_loader), 
            total=len(query_loader), 
            disable=self.verbose is False
        ):
            texts = [q.text for q in queries]
            with torch.no_grad():
                embeddings = model(texts)
            all_embeddings.append(embeddings.cpu().numpy())
            all_query_ids.extend(queries)
            
            if len(all_embeddings) == chroma_batch_size or it == len(query_loader) - 1:
                all_embeddings = np.vstack(all_embeddings)

                sem_search_results = collection.query(
                    query_embeddings=all_embeddings,
                    n_results=topk - 1,                 # for some reason, it returns one more doc...
                    include=["metadatas", "distances"]
                )
                doc_ids = [
                    [doc["doc_id"] for doc in q_docs] 
                    for q_docs in sem_search_results["metadatas"]
                ]

                for query, docs, distances in zip(
                    all_query_ids, doc_ids, sem_search_results["distances"]
                ):
                    query_id = (
                        f"query_{len(all_results)}" 
                        if query.id is None 
                        else query.id
                    )
                    all_results.append(SemanticSearchResult(
                        query_id=query_id,
                        doc_ids=docs,
                        distances=distances
                    ))
                
                all_embeddings = []
                all_query_ids = []
        
        if was_training:
            model.train()

        if save_dirpath is not None:
            LocalTopKDocumentsStore(save_dirpath, topk=topk).store_topk_documents(
                all_results
            )
        return all_results
    
    def _retrieve_documents_from_result_set(
        self, result_set: list[dict], document_collection_name: str
    ) -> list[dict]:
        all_docs = []
        col = self.client.get_collection(document_collection_name)

        for results in result_set:
            doc_ids = results["doc_ids"]
            revert_indices = np.argsort(
                pd.Series(doc_ids).sort_values().index
            )
            response = col.get(doc_ids)["metadatas"]
            documents = [
                json.loads(meta["json_string"])
                for meta in np.array(response)[revert_indices]
            ]
            all_docs.append({
                "query_id": results["query_id"],
                "docs": documents
            })

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

    def retrieve_topk_documents(
        self, model: EmbeddingModel, query_loader: DataLoader, topk: int = 10,
        save_dirpath: str | None = None, load_dirpath: str | None = None,
    ) -> list[SemanticSearchResult]:
        if load_dirpath is not None:
            try:
                topk_store = LocalTopKDocumentsStore(load_dirpath, topk=topk)
                return topk_store.load_topk_documents(query_loader)
            except:
                pass

        if self.vector_store_in_memory is None:
            self.vector_store_in_memory = self._load_embeddings()        
        all_document_ids = np.array([
            file[:file.rfind(".")]
            for file in sorted(os.listdir(self.save_dirpath))
        ])

        all_results_sets = []
        all_queries = []
        for queries in tqdm(query_loader):
            texts = [q.text for q in queries]
            with torch.no_grad():
                query_emb = model(texts)
    
            all_results_sets.extend(semantic_search(
                query_emb, self.vector_store_in_memory, query_chunk_size=100, 
                corpus_chunk_size=10_000, top_k=topk
            ))
            all_queries.extend(queries)
    
        all_results = []
        for db_matches, query in zip(all_results_sets, all_queries):
            db_indices = [db_match["corpus_id"] for db_match in db_matches]
            db_scores = [db_match["score"] for db_match in db_matches]

            query_id = (
                f"query_{len(all_results)}"
                if query.id is None
                else query.id
            )
            all_results.append(SemanticSearchResult(
                query_id=query_id,
                doc_ids=all_document_ids[db_indices].tolist(),
                distances=(1 - np.array(db_scores)).tolist()
            ))

        if save_dirpath is not None:
            LocalTopKDocumentsStore(save_dirpath, topk=topk).store_topk_documents(
                all_results
            )
        return all_results

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


class LocalTopKDocumentsStore:
    def __init__(self, dirpath: str, topk: int) -> None:
        self.dirpath = dirpath
        self.topk = topk
        
    def store_topk_documents(
        self, sem_search_results: list[SemanticSearchResult]
    ) -> None:
        os.makedirs(self.dirpath, exist_ok=True)

        for query_results in sem_search_results:
            docs_to_save = [
                { "doc_id": doc_id, "distance": dist }
                for doc_id, dist in zip(
                    query_results.doc_ids, query_results.distances
                )
            ]
    
            path = os.path.join(self.dirpath, f"{query_results.query_id}.json")
            with open(path, "w") as f:
                json.dump(docs_to_save, f, ensure_ascii=False)
    
    def load_topk_documents(
        self, query_loader: DataLoader
    ) -> list[SemanticSearchResult]:
        available_query_ids = [
            filename[:filename.rfind(".")]
            for filename in sorted(os.listdir(self.dirpath))
        ]
        requested_query_ids = [
            query.id
            for query in query_loader.dataset.queries
        ]
        if (np.isin(requested_query_ids, available_query_ids) == False).sum() > 0:
            raise ValueError(
                "Not all requested top K documents for each are stored locally"
            )
        
        topk_documents: list[SemanticSearchResult] = []
        for query_id in requested_query_ids:
            path = os.path.join(self.dirpath, f"{query_id}.json")
            with open(path) as f:
                data = json.load(f)
            
            topk_documents.append(SemanticSearchResult(
                query_id=query_id,
                doc_ids=[d["doc_id"] for d in data],
                distances=[d["distance"] for d in data]
            ))
            
        return topk_documents
    

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
    text_dirpath = "data/texts"

    model = ModelSetup._setup_gte_large(model_max_length=4096)


    store = Chroma_EmbeddingStore(client, verbose=True)
    collection_name = "embeddings-gte_large-simple-v0"

    # compute_embeddings_wrapper(client, model, text_dirpath, collection_name)

    # Perform semantic search    
    QUERIES = [
        { "text": "I want a dataset about movies reviews" }, 
        { "text": "Second query inbound" } 
    ]
    ds = Queries(queries=QUERIES)
    query_loader = DataLoader(ds, batch_size=2, collate_fn=lambda x: x)

    batch = next(iter(query_loader))

    exit()

    results = store.retrieve_topk_documents(
        model, query_loader, topk=10
    )

    