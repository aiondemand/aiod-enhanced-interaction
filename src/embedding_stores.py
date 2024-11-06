import json
import pandas as pd
import os
import sys
from torch.utils.data import DataLoader
from chromadb.api.client import Client as ChromaClient
from pymilvus import CollectionSchema, DataType, FieldSchema, MilvusClient
from pymilvus.milvus_client import IndexParams
from chromadb import Collection
from tqdm import tqdm
import torch
import uuid
import numpy as np
from abc import ABC, abstractmethod 
from sentence_transformers.util import semantic_search
from langchain_community.callbacks import get_openai_callback

from lang_chains import SimpleChain
from llm_metadata_filter import DatasetMetadataTemplate, LLM_MetadataExtractor, build_milvus_filter, apply_lowercase
from model.base import EmbeddingModel
import utils
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
        save_dirpath: str | None = None, load_dirpaths: str | list[str] | None = None,
        **kwargs
    ) -> list[SemanticSearchResult]:
        pass

    @abstractmethod
    def translate_sem_results_to_documents(
        self, result_set: list[SemanticSearchResult], **kwargs
    ) -> list[RetrievedDocuments]:
        pass
        

class Milvus_EmbeddingStore(EmbeddingStore):
    def __init__(
        self, client: MilvusClient, 
        emb_dimensionality: int, 
        chunk_embedding_store: bool = False, 
        extract_metadata: bool = False,
        verbose: bool = False
    ) -> None:
        self.client = client
        self.emb_dimensionality = emb_dimensionality
        self.chunk_embedding_store = chunk_embedding_store
        self.extract_metadata = extract_metadata
        self.verbose = verbose

    def _create_collection(self, collection_name: str) -> None:
        if self.client.has_collection(collection_name) is False:
            schema = self.client.create_schema(auto_id=True)
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=1024)
            schema.add_field("doc_id", DataType.VARCHAR, max_length=20)
            
            if self.extract_metadata:
                # metadata for filtering purposes
                schema.add_field("platform", DataType.VARCHAR, max_length=20)
                schema.add_field("date_published", DataType.VARCHAR, max_length=20)
                schema.add_field("year", DataType.INT16)
                schema.add_field("month", DataType.INT16)
                schema.add_field(
                    "domains", 
                    DataType.ARRAY, 
                    element_type=DataType.VARCHAR, 
                    max_length=30, 
                    max_capacity=5,
                    default=None
                )
                schema.add_field(
                    "task_types", 
                    DataType.ARRAY, 
                    element_type=DataType.VARCHAR, 
                    max_length=50, 
                    max_capacity=20,
                    default=None
                )
                schema.add_field("license", DataType.VARCHAR, max_length=10, default=None)

                schema.add_field("size_in_mb", DataType.FLOAT, default=None)
                schema.add_field("num_datapoints", DataType.INT64, default=None)
                schema.add_field("size_category", DataType.VARCHAR, max_length=20, default=None)
                schema.add_field(
                    "modality", 
                    DataType.ARRAY, 
                    element_type=DataType.VARCHAR, 
                    max_length=20, 
                    max_capacity=5,
                    default=None
                )
                schema.add_field(
                    "data_format", 
                    DataType.ARRAY, 
                    element_type=DataType.VARCHAR, 
                    max_length=10, 
                    max_capacity=10,
                    default=None
                )
                schema.add_field(
                    "languages", 
                    DataType.ARRAY, 
                    element_type=DataType.VARCHAR, 
                    max_length=5, 
                    max_capacity=50,
                    default=None
                )
            schema.verify()

            index_params = IndexParams()
            index_params.add_index("vector", "", "", metric_type="COSINE")
            index_params.add_index("doc_id", "", "")

            self.client.create_collection(
                collection_name=collection_name,
                dimension=self.emb_dimensionality,
                auto_id=True,
            )
            
    def store_embeddings(
        self, model: EmbeddingModel, loader: DataLoader, 
        collection_name: str, milvus_batch_size: int = 50,
        extract_metadata_llm: LLM_MetadataExtractor | None = None
    ) -> None:
        was_training = model.training
        model.eval()
        self._create_collection(collection_name)
    
        all_embeddings = []
        all_ids = []
        all_doc_ids = []
        all_metadatas = []

        for it, (texts, doc_ids) in tqdm(
            enumerate(loader), total=len(loader), disable=self.verbose is False
        ):
            with torch.no_grad():
                chunks_embeddings_of_multiple_docs = model(texts)
            if chunks_embeddings_of_multiple_docs[0].ndim == 1:
                chunks_embeddings_of_multiple_docs = [emb[None] for emb in chunks_embeddings_of_multiple_docs]

            docs_metadata = [{} for _ in range(len(doc_ids))]
            if extract_metadata_llm is not None:
                docs_metadata = [
                    apply_lowercase(extract_metadata_llm(t))
                    for t in texts
                ]

            for chunk_embeds_of_a_doc, doc_id, metadata in zip(
                chunks_embeddings_of_multiple_docs, doc_ids, docs_metadata
            ):
                all_embeddings.extend([
                    chunk_emb for chunk_emb in chunk_embeds_of_a_doc.cpu().numpy()
                ])
                all_ids.extend([
                    str(uuid.uuid4()) for _ in range(len(chunk_embeds_of_a_doc))
                ])
                all_doc_ids.extend([doc_id] * len(chunk_embeds_of_a_doc))
                all_metadatas.extend([metadata] * len(chunk_embeds_of_a_doc))
    
            if (len(all_embeddings) >= milvus_batch_size or it == len(loader) - 1):
                data = [
                    {
                        "vector": emb,
                        "doc_id": doc_id,
                        **metadata
                    }
                    for emb, doc_id, metadata in zip(all_embeddings, all_doc_ids, all_metadatas)
                ]
                self.client.insert(collection_name=collection_name, data=data)
                
                all_embeddings = []
                all_ids = []
                all_doc_ids = []
                all_metadatas = []

        if was_training:
            model.train()

    def retrieve_topk_document_ids(
        self, model: EmbeddingModel, query_loader: DataLoader, topk: int = 10, 
        save_dirpath: str | None = None, load_dirpaths: str | list[str] | None = None,
        emb_collection_name: str | None = None, milvus_batch_size: int = 50,
        extract_conditions_llm: LLM_MetadataExtractor | None = None
    ) -> list[SemanticSearchResult]: 
        if load_dirpaths is not None:
            try:
                topk_store = LocalTopKDocumentsStore(topk=topk)
                return topk_store.load_topk_documents(query_loader, load_dirpaths)
            except:
                pass

        was_training = model.training
        model.eval()
        if self.client.has_collection(emb_collection_name) is False:
            raise ValueError(f"Collection '{emb_collection_name}' doesnt exist")

        all_results = []
        all_embeddings = []
        all_queries = []
        
        for it, queries in tqdm(
            enumerate(query_loader), 
            total=len(query_loader), 
            disable=self.verbose is False
        ):
            texts = [q.text for q in queries]

            filter_strings = None
            if extract_conditions_llm is not None:
                query_metadata = [
                    apply_lowercase(extract_conditions_llm(t))
                    for t in texts
                ]
                filter_strings = [
                    build_milvus_filter(meta) 
                    for meta in query_metadata
                ]
                
            with torch.no_grad():
                query_embeddings = model(texts)
            if query_embeddings[0].ndim == 2:
                if sum([len(q_emb) != 1 for q_emb in query_embeddings]) > 0:
                    raise ValueError("We dont support queries that consist of multiple chunks")
                query_embeddings = [q_emb[0] for q_emb in query_embeddings]

            all_embeddings.extend(q_emb.cpu().numpy() for q_emb in query_embeddings)
            all_queries.extend(queries)
            
            if (len(all_embeddings) >= milvus_batch_size or it == len(query_loader) - 1):
                all_embeddings = np.stack(all_embeddings).tolist()

                # TODO check
                if filter_strings is None:
                    sem_search_results = list(self.client.search(
                        collection_name=emb_collection_name,
                        data=all_embeddings,
                        limit=topk * 10 if self.chunk_embedding_store else topk+1,
                        output_fields=["doc_id"]
                    ))
                else:
                    sem_search_results = [
                        list(self.client.search(
                            collection_name=emb_collection_name,
                            data=[emb],
                            filter=filter,
                            limit=topk * 10 if self.chunk_embedding_store else topk+1,
                            output_fields=["doc_id"]
                        )[0])
                        for emb, filter in zip(all_embeddings, filter_strings)
                    ]

                for query, query_results in zip(all_queries, sem_search_results):
                    query_id = (
                        f"query_{len(all_results)}" 
                        if query.id is None 
                        else query.id
                    )

                    doc_ids = [match["entity"]["doc_id"] for match in query_results]
                    distances = [1 - match["distance"] for match in query_results] #be aware we work with similarities, not distances

                    indices = pd.Series(data=doc_ids).drop_duplicates().index.values[:topk]
                    filtered_docs = [doc_ids[idx] for idx in indices]
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
        # There is no collection containing the whole stringified JSON documents of 
        # assets, hence we cannot translate doc IDs to whole documents using vector 
        # database only
        pass


class Chroma_EmbeddingStore(EmbeddingStore):
    def __init__(
        self, client: ChromaClient, 
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
            collection = self._create_collection(collection_name)
        
        return collection

    def _create_collection(self, collection_name: str) -> Collection:
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
    
            if (len(all_embeddings) >= chroma_batch_size or it == len(loader) - 1):
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
            
            if (len(all_embeddings) >= chroma_batch_size or it == len(query_loader) - 1):
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
