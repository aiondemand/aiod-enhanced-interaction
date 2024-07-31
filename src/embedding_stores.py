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

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(src_dir)

from model.base import EmbeddingModel
import utils
from dataset import AIoD_Documents, Queries
from data_types import RetrievedDocuments, SemanticSearchResult


class EmbeddingStore(ABC):
    @abstractmethod
    def store_embeddings(
        self, model: EmbeddingModel, loader: DataLoader, **kwargs
    ) -> None:
        pass

    @abstractmethod
    def retrieve_topk_document_ids(
        self, model: EmbeddingModel, query_loader: DataLoader, topk: int = 10, 
        save_dirpath: str | None = None, load_dirpath: str | None = None, **kwargs
    ) -> list[SemanticSearchResult]:
        pass

    @abstractmethod
    def translate_sem_results_to_documents(
        self, result_set: list[SemanticSearchResult], **kwargs
    ) -> list[RetrievedDocuments]:
        pass
        

class Chroma_EmbeddingStore(EmbeddingStore):
    def __init__(
        self, client: Client, 
        chunk_embedding_store: bool = False, 
        verbose: bool = False
    ) -> None:
        self.client = client
        self.chunk_embedding_store = chunk_embedding_store
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
        self, model: EmbeddingModel, loader: DataLoader, 
        collection_name: str, chroma_batch_size: int = 50
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
                chunks_embeddings_of_multiple_docs = model(texts)
            if chunks_embeddings_of_multiple_docs[0].ndim == 1:
                chunks_embeddings_of_multiple_docs = [emb[None] for emb in chunks_embeddings_of_multiple_docs]

            for chunk_embeds_of_a_doc, doc_id in zip(chunks_embeddings_of_multiple_docs, doc_ids):
                all_embeddings.extend([
                    chunk_emb for chunk_emb in chunk_embeds_of_a_doc.cpu().numpy()
                ])
                all_ids.extend([
                    str(uuid.uuid4()) for _ in range(len(chunk_embeds_of_a_doc))
                ])
                all_meta.extend([
                    {"doc_id": doc_id} for _ in range(len(chunk_embeds_of_a_doc))
                ])
    
            if (
                it != 0 and len(all_embeddings) % chroma_batch_size == 0
                or it == len(loader) - 1
            ):
                all_embeddings = np.stack(all_embeddings)
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

    def retrieve_topk_document_ids(
        self, model: EmbeddingModel, query_loader: DataLoader, topk: int = 10, 
        save_dirpath: str | None = None, load_dirpaths: str | list[str] | None = None,
        emb_collection_name: str | None = None, chroma_batch_size: int = 50,
    ) -> list[SemanticSearchResult]:
        if load_dirpaths is not None:
            try:
                topk_store = LocalTopKDocumentsStore(topk=topk)
                return topk_store.load_topk_documents(query_loader, load_dirpaths)
            except:
                pass

        was_training = model.training
        model.eval()
        collection = self._get_collection(emb_collection_name)

        all_results = []
        all_embeddings = []
        all_queries = []
        
        for it, queries in tqdm(
            enumerate(query_loader), 
            total=len(query_loader), 
            disable=self.verbose is False
        ):
            texts = [q.text for q in queries]
            with torch.no_grad():
                query_embeddings = model(texts)
            if query_embeddings[0].ndim == 2:
                if sum([len(q_emb) != 1 for q_emb in query_embeddings]) > 0:
                    raise ValueError("We dont support queries that consist of multiple chunks")
                query_embeddings = [q_emb[0] for q_emb in query_embeddings]

            all_embeddings.extend(q_emb.cpu().numpy() for q_emb in query_embeddings)
            all_queries.extend(queries)
            
            if (
                it != 0 and it % chroma_batch_size == 0 
                or it == len(query_loader) - 1
            ):
                all_embeddings = np.stack(all_embeddings)

                sem_search_results = collection.query(
                    query_embeddings=all_embeddings,
                    n_results=topk * 10 if self.chunk_embedding_store else topk+1,
                    include=["metadatas", "distances"]
                )
                doc_ids = [
                    [doc["doc_id"] for doc in q_docs] 
                    for q_docs in sem_search_results["metadatas"]
                ]

                for query, docs, distances in zip(
                    all_queries, doc_ids, sem_search_results["distances"]
                ):
                    query_id = (
                        f"query_{len(all_results)}" 
                        if query.id is None 
                        else query.id
                    )
                    indices = pd.Series(data=docs).drop_duplicates().index.values[:topk]
                    filtered_docs = [docs[idx] for idx in indices]
                    filtered_distances = [distances[idx] for idx in indices]

                    all_results.append(SemanticSearchResult(
                        query_id=query_id,
                        doc_ids=filtered_docs,
                        distances=filtered_distances
                    ))
                
                all_embeddings = []
                all_queries = []
        
        if was_training:
            model.train()

        if save_dirpath is not None:
            topk_store = LocalTopKDocumentsStore(topk=topk)
            topk_store.store_topk_documents(all_results, save_dirpath)
        return all_results
    
    def translate_sem_results_to_documents(
        self, result_set: list[SemanticSearchResult], document_collection_name: str
    ) -> list[RetrievedDocuments]:
        all_docs = []
        col = self.client.get_collection(document_collection_name)

        for results in result_set:
            doc_ids = results.doc_ids
            revert_indices = np.argsort(
                pd.Series(doc_ids).sort_values().index
            )
            response = col.get(doc_ids)["metadatas"]
            documents = [
                json.loads(meta["json_string"])
                for meta in np.array(response)[revert_indices]
            ]
            all_docs.append(RetrievedDocuments(
                query_id=results.query_id,
                document_objects=documents
            ))

        return all_docs
    

# TODO this store doesnt support transformation of doc IDs to doc JSONs
# TODO this store doesnt support saving embeddings of multiple chunks of one document yet
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

    def retrieve_topk_document_ids(
        self, model: EmbeddingModel, query_loader: DataLoader, topk: int = 10,
        save_dirpath: str | None = None, load_dirpaths: str | list[str] | None = None,
    ) -> list[SemanticSearchResult]:
        if load_dirpaths is not None:
            try:
                topk_store = LocalTopKDocumentsStore(topk=topk)
                return topk_store.load_topk_documents(query_loader, load_dirpaths)
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
            topk_store = LocalTopKDocumentsStore(topk=topk)
            topk_store.store_topk_documents(all_results, save_dirpath)
        return all_results
    
    def translate_sem_results_to_documents(
        self, result_set: list[SemanticSearchResult]
    ) -> list[dict]:
        # TODO
        pass

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
    def __init__(self, topk: int) -> None:
        self.topk = topk
        
    def store_topk_documents(
        self, sem_search_results: list[SemanticSearchResult], save_dirpath: str
    ) -> None:
        os.makedirs(save_dirpath, exist_ok=True)

        for query_results in sem_search_results:
            docs_to_save = [
                { "doc_id": doc_id } for doc_id in query_results.doc_ids
            ]
            if query_results.distances is not None:
                for it, dist in enumerate(query_results.distances):
                    docs_to_save[it]["distance"] = dist
            
            path = os.path.join(save_dirpath, f"{query_results.query_id}.json")
            with open(path, "w") as f:
                json.dump(docs_to_save, f, ensure_ascii=False)
    
    def load_topk_documents(
        self, query_loader: DataLoader, load_dirpaths: str | list[str]
    ) -> list[SemanticSearchResult]:
        if type(load_dirpaths) is str:
            load_dirpaths = [load_dirpaths]

        available_query_ids_path_map = {}
        for path in load_dirpaths:
            available_query_ids_path_map.update({ 
                filename[:filename.rfind(".")]: path  
                for filename in sorted(os.listdir(path))
            })            
        available_query_ids = list(available_query_ids_path_map.keys())
        
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
            dirpath = available_query_ids_path_map[query_id]
            fullpath = os.path.join(dirpath, f"{query_id}.json")
            with open(fullpath) as f:
                data = json.load(f)
            
            topk_documents.append(SemanticSearchResult(
                query_id=query_id,
                doc_ids=[d["doc_id"] for d in data],
                distances=(
                    [d["distance"] for d in data]
                    if data[0].get("distance", None) is not None
                    else None
                )
            ))
            
        return topk_documents
