from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import pandas as pd
from pymilvus import DataType, MilvusClient
from pymilvus.milvus_client import IndexParams
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.config import settings
from app.schemas.enums import AssetType
from app.schemas.params import MilvusSearchParams
from app.schemas.search_results import SearchResults
from app.services.inference.model import AiModel


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
    def retrieve_topk_document_ids(self, search_params: MilvusSearchParams) -> SearchResults:
        pass

    @abstractmethod
    def get_asset_embeddings(
        self, asset_id: int, asset_type: AssetType
    ) -> list[list[float]] | None:
        pass


class MilvusEmbeddingStore(EmbeddingStore):
    # TODO
    # In the future we should pass an embedding_dim as an argument
    def __init__(
        self,
        verbose: bool = False,
    ) -> None:
        self.emb_dimensionality = 1024
        self.extract_metadata = settings.MILVUS.EXTRACT_METADATA
        self.chunk_embedding_store = settings.MILVUS.STORE_CHUNKS
        self.verbose = verbose

        self.client: MilvusClient | None = None

    @property
    def vector_index_kwargs(self) -> dict:
        return {
            "index_type": "HNSW_SQ",
            "metric_type": "COSINE",
            "params": {"sq_type": "SQ8"},
        }

    @property
    def scalar_index_kwargs(self) -> dict:
        return {"index_type": "INVERTED"}

    async def init() -> MilvusEmbeddingStore:
        obj = MilvusEmbeddingStore()
        await obj.init_connection()
        return obj

    async def init_connection(self) -> bool:
        for _ in range(5):
            try:
                self.client = MilvusClient(
                    uri=str(settings.MILVUS.URI), token=settings.MILVUS.MILVUS_TOKEN
                )
                return True
            except Exception:
                logging.warning("Failed to connect to Milvus vector database. Retrying...")
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

                    # TODO once we have arbitrary metadata fields, we should come up with some
                    # value restrictions (e.g., string max length, array max capacity, etc.)
                    schema.add_field(
                        "date_published", DataType.VARCHAR, max_length=22, nullable=True
                    )
                    schema.add_field("size_in_mb", DataType.FLOAT, nullable=True)
                    schema.add_field("license", DataType.VARCHAR, max_length=20, nullable=True)

                    schema.add_field(
                        "task_types",
                        DataType.ARRAY,
                        element_type=DataType.VARCHAR,
                        max_length=50,
                        max_capacity=100,
                        nullable=True,
                    )
                    schema.add_field(
                        "languages",
                        DataType.ARRAY,
                        element_type=DataType.VARCHAR,
                        max_length=2,
                        max_capacity=200,
                        nullable=True,
                    )
                    schema.add_field("datapoints_upper_bound", DataType.INT64, nullable=True)
                    schema.add_field("datapoints_lower_bound", DataType.INT64, nullable=True)

            schema.verify()

            index_params = IndexParams()
            index_params = self.client.prepare_index_params()

            index_params.add_index(field_name="vector", **self.vector_index_kwargs)
            index_params.add_index(field_name="doc_id", **self.scalar_index_kwargs)

            if self.extract_metadata:
                if asset_type == AssetType.DATASETS:
                    index_params.add_index(field_name="date_published", **self.scalar_index_kwargs)
                    index_params.add_index(field_name="size_in_mb", **self.scalar_index_kwargs)
                    index_params.add_index(field_name="license", **self.scalar_index_kwargs)
                    index_params.add_index(field_name="task_types", **self.scalar_index_kwargs)
                    index_params.add_index(field_name="languages", **self.scalar_index_kwargs)
                    index_params.add_index(
                        field_name="datapoints_upper_bound", **self.scalar_index_kwargs
                    )
                    index_params.add_index(
                        field_name="datapoints_lower_bound", **self.scalar_index_kwargs
                    )

            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
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
        **kwargs,
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
                    for emb, doc_id, meta in zip(all_embeddings, all_doc_ids, all_metadata)
                ]
                total_inserted += self.client.insert(collection_name=collection_name, data=data)[
                    "insert_count"
                ]

                all_embeddings = []
                all_doc_ids = []
                all_metadata = []

        return total_inserted

    def remove_embeddings(self, doc_ids: list[str], asset_type: AssetType) -> int:
        collection_name = self.get_collection_name(asset_type)

        return self.client.delete(collection_name, filter=f"doc_id in {doc_ids}")["delete_count"]

    def retrieve_topk_document_ids(self, search_params: MilvusSearchParams) -> SearchResults:
        collection_name = self.get_collection_name(search_params.asset_type)

        if self.client.has_collection(collection_name) is False:
            raise ValueError(f"Collection '{collection_name}' does not exist")
        self.client.load_collection(collection_name)

        query_results = list(
            self.client.search(collection_name=collection_name, **search_params.get_params())
        )

        doc_ids = []
        distances = []
        for results in query_results:
            doc_ids.extend([match["entity"]["doc_id"] for match in results])
            distances.extend([1 - match["distance"] for match in results])

        help_df = pd.DataFrame(data=[doc_ids, distances]).T
        help_df.columns = ["doc_ids", "distances"]
        indices = (
            help_df.sort_values(by=["distances"])
            .drop_duplicates(subset=["doc_ids"])
            .index.values[: search_params.topk]
        )

        return SearchResults(
            doc_ids=[doc_ids[idx] for idx in indices],
            distances=[distances[idx] for idx in indices],
        )

    def get_asset_embeddings(
        self, asset_id: int, asset_type: AssetType
    ) -> Optional[List[List[float]]]:
        collection_name = self.get_collection_name(asset_type)

        try:
            data = self.client.query(
                collection_name=collection_name,
                filter=f'doc_id == "{asset_id}"',
                output_fields=["vector"],
            )
            if not data:
                return None

            embeddings = [item["vector"] for item in data]
            return embeddings
        except Exception as e:
            logging.error(f"Failed to retrieve embeddings for doc_id '{asset_id}': {e}")
            return None
