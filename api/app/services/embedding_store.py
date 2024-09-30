from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from app.config import settings
from app.schemas.search_results import SearchResults
from app.services.inference.model import AiModel
from pymilvus import MilvusClient
from torch.utils.data import DataLoader
from tqdm import tqdm


class EmbeddingStore(ABC):
    @abstractmethod
    def store_embeddings(
        self, model: AiModel, loader: DataLoader, collection_name: str, **kwargs
    ) -> list[str]:
        pass

    @abstractmethod
    def exists_collection(self, collection_name: str) -> bool:
        pass

    @abstractmethod
    def get_all_document_ids(self, collection_name: str) -> list[str]:
        pass

    @abstractmethod
    def retrieve_topk_document_ids(
        self,
        model: AiModel,
        query_text: str,
        collection_name: str,
        topk: int = 10,
    ) -> SearchResults:
        pass


class Milvus_EmbeddingStore(EmbeddingStore):
    def __init__(self, verbose: bool = False) -> None:
        self.emb_dimensionality = settings.MILVUS.EMB_DIM
        self.chunk_embedding_store = settings.MILVUS.STORE_CHUNKS
        self.verbose = verbose

        self.client = None

    async def init() -> Milvus_EmbeddingStore:
        obj = Milvus_EmbeddingStore()
        await obj.init_connection()
        return obj

    async def init_connection(self) -> None:
        for _ in range(5):
            try:
                self.client = MilvusClient(
                    uri=settings.MILVUS.URI, token=settings.MILVUS.MILVUS_TOKEN
                )
                return True
            except Exception:  # TODO
                await asyncio.sleep(5)
        else:
            raise ValueError(
                "Connection to Milvus vector database has not been established"
            )

    def _create_collection(self, collection_name: str) -> None:
        if self.client.has_collection(collection_name) is False:
            self.client.create_collection(
                collection_name=collection_name,
                dimension=self.emb_dimensionality,
                auto_id=True,
            )

    def exists_collection(self, collection_name: str) -> bool:
        return self.client.has_collection(collection_name)

    def get_all_document_ids(self, collection_name: str) -> list[str]:
        if self.client.has_collection(collection_name) is False:
            return []

        data = list(
            self.client.query(
                collection_name=collection_name,
                filter="id > 0",
                output_fields=["doc_id"],
            )
        )
        all_doc_ids = [str(x["doc_id"]) for x in data]
        return np.unique(np.array(all_doc_ids)).tolist()

    def store_embeddings(
        self,
        model: AiModel,
        loader: DataLoader,
        collection_name: str,
        milvus_batch_size: int = 50,
    ) -> list[str]:
        self._create_collection(collection_name)

        all_embeddings = []
        all_doc_ids = []
        really_all_doc_ids = []
        for it, (texts, doc_ids) in tqdm(
            enumerate(loader), total=len(loader), disable=self.verbose is False
        ):
            chunks_embeddings_of_multiple_docs = model.compute_asset_embeddings(texts)

            for chunk_embeds_of_a_doc, doc_id in zip(
                chunks_embeddings_of_multiple_docs, doc_ids
            ):
                all_embeddings.extend(
                    [chunk_emb for chunk_emb in chunk_embeds_of_a_doc.cpu().numpy()]
                )
                all_doc_ids.extend([doc_id] * len(chunk_embeds_of_a_doc))

            if len(all_embeddings) >= milvus_batch_size or it == len(loader) - 1:
                data = [
                    {"vector": emb, "doc_id": doc_id}
                    for emb, doc_id in zip(all_embeddings, all_doc_ids)
                ]
                self.client.insert(collection_name=collection_name, data=data)
                really_all_doc_ids.extend(all_doc_ids)

                all_embeddings = []
                all_doc_ids = []

        return really_all_doc_ids

    def retrieve_topk_document_ids(
        self,
        model: AiModel,
        query_text: str,
        collection_name: str,
        topk: int = 10,
    ) -> SearchResults:
        if self.client.has_collection(collection_name) is False:
            raise ValueError(f"Collection '{collection_name}' doesnt exist")

        with torch.no_grad():
            query_embeddings = model.compute_query_embeddings([query_text])

        query_results = self.client.search(
            collection_name=collection_name,
            data=query_embeddings,
            limit=topk * 10 if self.chunk_embedding_store else topk + 1,
            output_fields=["doc_id"],
            search_params={"metric_type": "COSINE"},
        )[0]
        doc_ids = [match["entity"]["doc_id"] for match in query_results]
        distances = [1 - match["distance"] for match in query_results]

        indices = pd.Series(data=doc_ids).drop_duplicates().index.values[:topk]
        filtered_docs = [doc_ids[idx] for idx in indices]
        filtered_distances = [distances[idx] for idx in indices]

        return SearchResults(doc_ids=filtered_docs, distances=filtered_distances)
