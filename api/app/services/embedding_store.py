from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd
import torch
from app.config import settings
from app.schemas.query import QueueItem
from app.schemas.SemanticSearchResults import SemanticSearchResult
from app.services.inference.model import AiModel
from pymilvus import MilvusClient
from torch.utils.data import DataLoader
from tqdm import tqdm


class EmbeddingStore(ABC):
    @abstractmethod
    def store_embeddings(
        self, model: AiModel, loader: DataLoader, collection_name: str, **kwargs
    ) -> None:
        pass

    @abstractmethod
    def retrieve_topk_document_ids(
        self,
        model: AiModel,
        queue_item: QueueItem,
        collection_name: str,
        topk: int = 10,
    ) -> SemanticSearchResult:
        pass


class Milvus_EmbeddingStore(EmbeddingStore):
    _instance: Milvus_EmbeddingStore | None = None

    def __new__(cls) -> Milvus_EmbeddingStore:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.init()
        return cls._instance

    def init(self) -> None:
        db_opt = settings.MILVUS

        milvus_token = f"{db_opt.USER}:{db_opt.PASS}"
        self.client = MilvusClient(uri=db_opt.URI, token=milvus_token)

        self.emb_dimensionality = db_opt.EMB_DIM
        self.chunk_embedding_store = db_opt.STORE_CHUNKS
        self.verbose = True  # TODO

    def _create_collection(self, collection_name: str) -> None:
        if self.client.has_collection(collection_name) is False:
            self.client.create_collection(
                collection_name=collection_name,
                dimension=self.emb_dimensionality,
                auto_id=True,
            )

    def store_embeddings(
        self,
        model: AiModel,
        loader: DataLoader,
        collection_name: str,
        milvus_batch_size: int = 50,
    ) -> None:
        self._create_collection(collection_name)

        all_embeddings = []
        all_doc_ids = []
        for it, (texts, doc_ids) in tqdm(
            enumerate(loader), total=len(loader), disable=self.verbose is False
        ):
            with torch.no_grad():
                chunks_embeddings_of_multiple_docs = model(texts)
            if chunks_embeddings_of_multiple_docs[0].ndim == 1:
                chunks_embeddings_of_multiple_docs = [
                    emb[None] for emb in chunks_embeddings_of_multiple_docs
                ]

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

                all_embeddings = []
                all_doc_ids = []

    def retrieve_topk_document_ids(
        self,
        model: AiModel,
        queue_item: QueueItem,
        collection_name: str,
        topk: int = 10,
    ) -> SemanticSearchResult:
        if self.client.has_collection(collection_name) is False:
            raise ValueError(f"Collection '{collection_name}' doesnt exist")

        with torch.no_grad():
            query_embeddings = model.compute_embeddings([queue_item.query])

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

        return SemanticSearchResult(
            query_id=queue_item.id, doc_ids=filtered_docs, distances=filtered_distances
        )
