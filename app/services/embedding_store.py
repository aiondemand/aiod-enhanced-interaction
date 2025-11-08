from __future__ import annotations

import inspect
import json
import logging
import os
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Any, Generic, TypeVar
from uuid import uuid4
import numpy as np
import pandas as pd

from pymilvus import DataType, MilvusClient, MilvusUnavailableException, CollectionSchema
from torch.utils.data import DataLoader
from tqdm import tqdm

from app.config import settings
from app.schemas.enums import SupportedAssetType
from app.schemas.params import MilvusSearchParams, VectorSearchParams
from app.schemas.search_results import SearchResults
from app.services.inference.model import AiModel
from app.services.resilience import retry_loop

SearchParams = TypeVar("SearchParams", bound=VectorSearchParams)


class EmbeddingStore(Generic[SearchParams], ABC):
    @abstractmethod
    def create_search_params(self, **kwargs) -> SearchParams:
        raise NotImplementedError

    @abstractmethod
    def get_collection_name(self, asset_type: SupportedAssetType) -> str:
        raise NotImplementedError

    @abstractmethod
    def store_embeddings(
        self, model: AiModel, loader: DataLoader, asset_type: SupportedAssetType, **kwargs
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    def remove_embeddings(self, asset_ids: list[str], asset_type: SupportedAssetType) -> int:
        raise NotImplementedError

    @abstractmethod
    def exists_collection(self, asset_type: SupportedAssetType) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_all_asset_ids(self, asset_type: SupportedAssetType) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def retrieve_topk_asset_ids(self, search_params: SearchParams) -> SearchResults:
        raise NotImplementedError

    @abstractmethod
    def get_asset_embeddings(
        self, asset_id: str, asset_type: SupportedAssetType
    ) -> list[list[float]] | None:
        raise NotImplementedError


class MilvusClientResilientWrapper(MilvusClient):
    def __init__(self, uri: str, token: str | None = None) -> None:
        super().__init__(uri=uri, token=token)

    def __getattribute__(self, name: str) -> Any:
        attr = super().__getattribute__(name)

        if not callable(attr) or name.startswith("__"):
            return attr
        if "timeout" in inspect.signature(attr).parameters:
            attr = partial(attr, timeout=settings.MILVUS.TIMEOUT)
        return retry_loop(output_exception_cls=MilvusUnavailableException)(attr)


class MilvusEmbeddingStore(EmbeddingStore[MilvusSearchParams]):
    # TODO
    # In the future we should pass an embedding_dim as an argument
    def __init__(
        self,
        verbose: bool = False,
    ) -> None:
        self.emb_dimensionality = 1024
        self.chunk_embedding_store = settings.MILVUS.STORE_CHUNKS
        self.verbose = verbose
        self.metadata_field_config = self._load_metadata_field_config()

        try:
            self.client = MilvusClientResilientWrapper(
                uri=str(settings.MILVUS.URI), token=settings.MILVUS.MILVUS_TOKEN
            )
        except Exception as e:
            logging.error(e)
            logging.error("Milvus is unavailable. Application is being terminated now")
            os._exit(1)

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

    def create_search_params(self, **kwargs) -> MilvusSearchParams:
        return MilvusSearchParams(**kwargs)

    def get_collection_name(self, asset_type: SupportedAssetType) -> str:
        return f"{settings.MILVUS.COLLECTION_PREFIX}_{asset_type.value}"

    def _create_collection(self, asset_type: SupportedAssetType) -> None:
        collection_name = self.get_collection_name(asset_type)

        if self.client.has_collection(collection_name) is False:
            schema = self.client.create_schema(auto_id=True)
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("vector", DataType.FLOAT_VECTOR, dim=1024)
            schema.add_field("asset_id", DataType.VARCHAR, max_length=50)

            if settings.extracts_metadata_from_asset(asset_type):
                self._add_metadata_fields(schema, asset_type)

            schema.verify()

            index_params = self.client.prepare_index_params()

            index_params.add_index(field_name="vector", **self.vector_index_kwargs)
            index_params.add_index(field_name="asset_id", **self.scalar_index_kwargs)

            # TODO This has been intentionally commented out due to unexpected Milvus behavior
            # when trying to index scalar fields.... More information can be found here:
            # - https://github.com/aiondemand/aiod-enhanced-interaction/issues/77

            # if self.extract_metadata:
            #     if asset_type == AssetType.DATASETS:
            #         index_params.add_index(field_name="date_published", **self.scalar_index_kwargs)
            #         index_params.add_index(field_name="size_in_mb", **self.scalar_index_kwargs)
            #         index_params.add_index(field_name="license", **self.scalar_index_kwargs)
            #         index_params.add_index(field_name="task_types", **self.scalar_index_kwargs)
            #         index_params.add_index(field_name="languages", **self.scalar_index_kwargs)
            #         index_params.add_index(
            #             field_name="datapoints_upper_bound", **self.scalar_index_kwargs
            #         )
            #         index_params.add_index(
            #             field_name="datapoints_lower_bound", **self.scalar_index_kwargs
            #         )

            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params,
            )

    def _load_metadata_field_config(self) -> dict:
        config_path = Path("app/data/milvus_metadata_fields.json")
        if not config_path.exists():
            raise ValueError("Incorrect path to the Milvus Metadata Fields file")
        with config_path.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        return config

    def _metadata_fields_for(self, asset_type: SupportedAssetType) -> list[dict[str, Any]]:
        base_fields = self.metadata_field_config.get("base", [])
        type_fields = self.metadata_field_config.get(asset_type.value, [])
        return [*base_fields, *type_fields]

    def _add_metadata_fields(
        self, schema: CollectionSchema, asset_type: SupportedAssetType
    ) -> None:
        for field in self._metadata_fields_for(asset_type):
            try:
                self._add_metadata_field(schema, field)
            except Exception as exc:
                logging.warning(
                    "Failed to register Milvus metadata field '%s' for asset type '%s': %s",
                    field.get("field_name"),
                    asset_type.value,
                    exc,
                )

    @staticmethod
    def _resolve_data_type(data_type_name: str) -> DataType:
        try:
            return getattr(DataType, data_type_name)
        except AttributeError as exc:
            raise ValueError(f"Unsupported Milvus data type: {data_type_name}") from exc

    def _add_metadata_field(
        self, schema: CollectionSchema, field_definition: dict[str, Any]
    ) -> None:
        if "field_name" not in field_definition or "data_type" not in field_definition:
            raise ValueError("Field configuration must contain 'field_name' and 'data_type'")

        field_name = field_definition["field_name"]
        data_type = self._resolve_data_type(field_definition["data_type"])

        field_kwargs: dict[str, Any] = {
            key: value
            for key, value in field_definition.items()
            if key not in {"field_name", "data_type", "element_type"}
        }

        if "element_type" in field_definition:
            field_kwargs["element_type"] = self._resolve_data_type(field_definition["element_type"])

        schema.add_field(field_name, data_type, **field_kwargs)

    def exists_collection(self, asset_type: SupportedAssetType) -> bool:
        return self.client.has_collection(self.get_collection_name(asset_type))

    def get_all_asset_ids(self, asset_type: SupportedAssetType) -> list[str]:
        collection_name = self.get_collection_name(asset_type)

        if self.client.has_collection(collection_name) is False:
            return []
        self.client.load_collection(collection_name)

        data = list(
            self.client.query(
                collection_name=collection_name,
                filter="id > 0",
                output_fields=["asset_id"],
            )
        )
        all_asset_ids = [x["asset_id"] for x in data]
        return np.unique(np.array(all_asset_ids)).tolist()

    def store_embeddings(
        self,
        model: AiModel,
        loader: DataLoader,
        asset_type: SupportedAssetType,
        milvus_batch_size: int = 50,
        **kwargs,
    ) -> int:
        collection_name = self.get_collection_name(asset_type)
        self._create_collection(asset_type)

        all_embeddings: list[list[float]] = []
        all_asset_ids: list[str] = []
        all_metadata: list[dict] = []

        total_inserted = 0
        for it, (texts, asset_ids, assets_metadata) in tqdm(
            enumerate(loader), total=len(loader), disable=self.verbose is False
        ):
            chunks_embeddings_of_multiple_assets = model.compute_asset_embeddings(texts)
            for chunk_embeds_of_an_asset, asset_id, meta in zip(
                chunks_embeddings_of_multiple_assets, asset_ids, assets_metadata
            ):
                all_embeddings.extend(
                    [chunk_emb for chunk_emb in chunk_embeds_of_an_asset.cpu().numpy()]
                )
                all_asset_ids.extend([asset_id] * len(chunk_embeds_of_an_asset))
                all_metadata.extend([meta] * len(chunk_embeds_of_an_asset))

            if len(all_embeddings) >= milvus_batch_size or it == len(loader) - 1:
                data: list[dict] = [
                    {"vector": emb, "asset_id": asset_id, **meta}
                    for emb, asset_id, meta in zip(all_embeddings, all_asset_ids, all_metadata)
                ]
                total_inserted += self._insert_records(collection_name, data)

                # Store data locally into JSON files as well if we wish to do so
                # Used for storing cold start data in JSON format
                if settings.AIOD.STORE_DATA_IN_JSON and settings.AIOD.JSON_SAVEPATH is not None:
                    for i in range(len(data)):
                        data[i]["vector"] = data[i]["vector"].tolist()

                    full_json_filepath = (
                        settings.AIOD.JSON_SAVEPATH
                        / f"jsons/{collection_name}"
                        / f"{str(uuid4())}.json"
                    )
                    full_json_filepath.parent.mkdir(parents=True, exist_ok=True)

                    with open(full_json_filepath, "w") as f:
                        json.dump(data, f)

                all_embeddings = []
                all_asset_ids = []
                all_metadata = []

        return total_inserted

    def _insert_records(self, collection_name: str, data: list[dict]) -> int:
        try:
            self.client.insert(collection_name=collection_name, data=data)
            return len(data)
        except Exception:
            logging.warning("Failed to insert Milvus batch. Attempting per-row insert.")

        inserted = 0
        for row in data:
            try:
                self.client.insert(collection_name=collection_name, data=[row])
                inserted += 1
            except Exception:
                logging.warning(
                    "Failed to insert Milvus row for asset '%s' with metadata. Retrying without metadata.",
                    row.get("asset_id"),
                )
                row_no_metadata = {"vector": row["vector"], "asset_id": row["asset_id"]}

                self.client.insert(collection_name=collection_name, data=[row_no_metadata])
                inserted += 1

        return inserted

    def remove_embeddings(self, asset_ids: list[str], asset_type: SupportedAssetType) -> int:
        collection_name = self.get_collection_name(asset_type)

        return self.client.delete(collection_name, filter=f"asset_id in {asset_ids}")[
            "delete_count"
        ]

    def retrieve_topk_asset_ids(self, search_params: MilvusSearchParams) -> SearchResults:
        collection_name = self.get_collection_name(search_params.asset_type)

        if self.client.has_collection(collection_name) is False:
            raise ValueError(f"Collection '{collection_name}' does not exist")
        self.client.load_collection(collection_name)

        query_results = list(
            self.client.search(collection_name=collection_name, **search_params.get_params())
        )

        asset_ids: list[str] = []
        distances: list[float] = []
        for results in query_results:
            asset_ids.extend([match["entity"]["asset_id"] for match in results])
            distances.extend([1 - match["distance"] for match in results])

        help_df = pd.DataFrame(data=[asset_ids, distances]).T
        help_df.columns = ["asset_ids", "distances"]
        indices = (
            help_df.sort_values(by=["distances"])
            .drop_duplicates(subset=["asset_ids"])
            .index.values[: search_params.topk]
        )

        return SearchResults(
            asset_ids=[asset_ids[idx] for idx in indices],
            distances=[distances[idx] for idx in indices],
            asset_types=[search_params.asset_type for _ in indices],
        )

    def get_asset_embeddings(
        self, asset_id: str, asset_type: SupportedAssetType
    ) -> list[list[float]] | None:
        collection_name = self.get_collection_name(asset_type)

        try:
            data = self.client.query(
                collection_name=collection_name,
                filter=f"asset_id == '{asset_id}'",
                output_fields=["vector"],
            )
            if not data:
                return None

            embeddings = [item["vector"] for item in data]
            return embeddings
        except Exception as e:
            logging.error(f"Failed to retrieve embeddings for asset_id '{asset_id}': {e}")
            return None
