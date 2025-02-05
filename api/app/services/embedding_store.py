from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch
from app.config import settings
from app.schemas.enums import AssetType
from app.schemas.search_results import SearchResults
from app.services.inference.model import AiModel
from pymilvus import DataType, MilvusClient
from pymilvus.milvus_client import IndexParams
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, List


class EmbeddingStore(ABC):
    @abstractmethod
    def get_collection_name(self, asset_type: AssetType) -> str:
        pass

    @abstractmethod
    def store_embeddings(
        self, model: AiModel, loader: DataLoader, asset_type: AssetType, **kwargs
    ) -> int:
        pass

    @abstractmethod
    def remove_embeddings(self, doc_ids: list[str], asset_type: AssetType) -> int:
        pass

    @abstractmethod
    def exists_collection(self, asset_type: AssetType) -> bool:
        pass

    @abstractmethod
    def get_all_document_ids(self, asset_type: AssetType) -> list[str]:
        pass

    @abstractmethod
    def retrieve_topk_document_ids(
        self,
        model: AiModel,
        query_text: str,
        asset_type: AssetType,
        topk: int = 10,
        filter: str = "",
        precomputed_embedding: Optional[List[float]] = None,
    ) -> SearchResults:
        pass

    @abstractmethod
    def get_embeddings(
        self, doc_id: str, collection_name: str
    ) -> Optional[List[List[float]]]:
        pass


class MilvusEmbeddingStore(EmbeddingStore):
    def __init__(
        self,
        verbose: bool = False,
    ) -> None:
        self.emb_dimensionality = 1024
        self.extract_metadata = settings.MILVUS.EXTRACT_METADATA
        self.chunk_embedding_store = settings.MILVUS.STORE_CHUNKS
        self.verbose = verbose

        self.client = None

    async def init() -> MilvusEmbeddingStore:
        obj = MilvusEmbeddingStore()
        await obj.init_connection()
        return obj

    async def init_connection(self) -> None:
        for _ in range(5):
            try:
                self.client = MilvusClient(
                    uri=str(settings.MILVUS.URI), token=settings.MILVUS.MILVUS_TOKEN
                )
                return True
            except Exception:
                logging.warning(
                    "Failed to connect to Milvus vector database. Retrying..."
                )
                await asyncio.sleep(5)
        else:
            err_msg = "Connection to Milvus vector database has not been established"
            logging.error(err_msg)
            raise ValueError(err_msg)

    def get_collection_name(self, asset_type: AssetType) -> str:
        return f"{settings.MILVUS.COLLECTION_PREFIX}_{asset_type.value}"

    def _create_collection(self, asset_type: AssetType) -> None:
        collection_name = self.get_collection_name(asset_type)

        if self.client.has_collection(collection_name) is False:
            schema = self.client.create_schema(auto_id=True)
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=1024)
            schema.add_field("doc_id", DataType.VARCHAR, max_length=20)

            if self.extract_metadata:
                if asset_type == AssetType.DATASETS:
                    # TODO
                    # Currently this schema reflects some what easily accessible and constant
                    # metadata we can retrieve from HuggingFace
                    schema.add_field("date_published", DataType.VARCHAR, max_length=22)
                    schema.add_field("size_in_mb", DataType.FLOAT, default=None)
                    schema.add_field(
                        "license", DataType.VARCHAR, max_length=20, default=None
                    )

                    schema.add_field(
                        "task_types",
                        DataType.ARRAY,
                        element_type=DataType.VARCHAR,
                        max_length=50,
                        max_capacity=20,
                        default=None,
                    )
                    schema.add_field(
                        "languages",
                        DataType.ARRAY,
                        element_type=DataType.VARCHAR,
                        max_length=2,
                        max_capacity=50,
                        default=None,
                    )
                    schema.add_field("datapoints_upper_bound", DataType.INT64)
                    schema.add_field("datapoints_lower_bound", DataType.INT64)

            schema.verify()

            index_params = IndexParams()
            index_params.add_index("vector", "", "", metric_type="COSINE")
            index_params.add_index("doc_id", "", "")

            self.client.create_collection(
                collection_name=collection_name,
                dimension=self.emb_dimensionality,
                auto_id=True,
            )

    def exists_collection(self, asset_type: AssetType) -> bool:
        return self.client.has_collection(self.get_collection_name(asset_type))

    def get_all_document_ids(self, asset_type: AssetType) -> list[str]:
        collection_name = self.get_collection_name(asset_type)

        if self.client.has_collection(collection_name) is False:
            return []
        self.client.load_collection(collection_name)

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
        asset_type: AssetType,
        milvus_batch_size: int = 50,
    ) -> int:
        collection_name = self.get_collection_name(asset_type)
        self._create_collection(asset_type)

        all_embeddings = []
        all_doc_ids = []
        all_metadata = []

        total_inserted = 0
        for it, (texts, doc_ids, docs_metadata) in tqdm(
            enumerate(loader), total=len(loader), disable=self.verbose is False
        ):
            chunks_embeddings_of_multiple_docs = model.compute_asset_embeddings(texts)
            for chunk_embeds_of_a_doc, doc_id, meta in zip(
                chunks_embeddings_of_multiple_docs, doc_ids, docs_metadata
            ):
                all_embeddings.extend(
                    [chunk_emb for chunk_emb in chunk_embeds_of_a_doc.cpu().numpy()]
                )
                all_doc_ids.extend([doc_id] * len(chunk_embeds_of_a_doc))
                all_metadata.extend([meta] * len(chunk_embeds_of_a_doc))

            if len(all_embeddings) >= milvus_batch_size or it == len(loader) - 1:
                data = [
                    {"vector": emb, "doc_id": doc_id, **meta}
                    for emb, doc_id, meta in zip(
                        all_embeddings, all_doc_ids, all_metadata
                    )
                ]
                total_inserted += self.client.insert(
                    collection_name=collection_name, data=data
                )["insert_count"]

                all_embeddings = []
                all_doc_ids = []
                all_metadata = []

        return total_inserted

    def remove_embeddings(self, doc_ids: list[str], asset_type: AssetType) -> int:
        collection_name = self.get_collection_name(asset_type)

        return self.client.delete(collection_name, filter=f"doc_id in {doc_ids}")[
            "delete_count"
        ]

    def retrieve_topk_document_ids(
        self,
        model: AiModel,
        query_text: str,
        asset_type: AssetType,
        topk: int = 10,
        filter: str = "",
        precomputed_embedding: Optional[List[float]] | None = None,
    ) -> SearchResults:
        collection_name = self.get_collection_name(asset_type)

        if self.client.has_collection(collection_name) is False:
            raise ValueError(f"Collection '{collection_name}' doesnt exist")
        self.client.load_collection(collection_name)

        if precomputed_embedding is None:
            if query_text is None:
                raise ValueError(
                    "Either query_text or precomputed_embedding must be provided."
                )
            if model is None:
                raise ValueError(
                    "AiModel instance must be provided to compute embeddings from query_text."
                )
            with torch.no_grad():
                query_embedding = model.compute_query_embeddings([query_text])
        else:
            query_embedding = [precomputed_embedding]

        query_results = self.client.search(
            collection_name=collection_name,
            data=query_embedding,
            limit=topk * 10 if self.chunk_embedding_store else topk + 1,
            output_fields=["doc_id"],
            search_params={"metric_type": "COSINE"},
            filter=filter,
        )[0]

        if not query_results:
            return SearchResults(doc_ids=[], distances=[])

        doc_ids = [match["entity"]["doc_id"] for match in query_results]
        distances = [1 - match["distance"] for match in query_results]

        indices = pd.Series(data=doc_ids).drop_duplicates().index.values[:topk]
        filtered_docs = [doc_ids[idx] for idx in indices]
        filtered_distances = [distances[idx] for idx in indices]

        return SearchResults(doc_ids=filtered_docs, distances=filtered_distances)

    def get_embeddings(
        self, doc_id: str, collection_name: str
    ) -> Optional[List[List[float]]]:

        self.client.load_collection(collection_name)

        try:
            data = self.client.query(
                collection_name=collection_name,
                filter=f'doc_id == "{doc_id}"',
                output_fields=["vector"],
            )
            if not data:
                return None

            embeddings = [item["vector"] for item in data]
            return embeddings
        except Exception as e:
            logging.error(f"Failed to retrieve embeddings for doc_id '{doc_id}': {e}")
            return None
