import json
import os
from argparse import ArgumentParser

import numpy as np
from pydantic import BaseModel, field_validator
from pymilvus import DataType, MilvusClient
from tqdm import tqdm


class InputArgs(BaseModel):
    input_dirpath: str
    uri: str
    username: str
    password: str
    metadata: bool

    @field_validator("metadata", mode="before")
    @classmethod
    def str_to_bool(cls, value: str) -> bool:
        if value.lower() not in ["true", "false"]:
            raise ValueError("Invalid value for boolean attribute")
        return value.lower() == "true"


def populate_database(args: InputArgs) -> None:
    if os.path.exists(args.input_dirpath) is False:
        exit(1)

    client = MilvusClient(uri=args.uri, user=args.username, password=args.password)

    for collection_name in sorted(os.listdir(args.input_dirpath)):
        path = os.path.join(args.input_dirpath, collection_name)
        populate_collection(client, collection_name, path, args.metadata)


def get_all_asset_ids(client: MilvusClient, collection_name: str) -> set[str]:
    client.load_collection(collection_name)

    data = list(
        client.query(
            collection_name=collection_name,
            filter="id > 0",
            output_fields=["asset_id"],
        )
    )
    all_asset_ids = [x["asset_id"] for x in data]
    return set(np.unique(np.array(all_asset_ids)).tolist())


def populate_collection(
    client: MilvusClient,
    collection_name: str,
    json_dirpath: str,
    extract_metadata: bool,
) -> None:
    unique_asset_ids = set()
    newly_created_col = create_new_collection(client, collection_name, extract_metadata)
    if newly_created_col is False:
        unique_asset_ids = get_all_asset_ids(client, collection_name)

    print(f"Populating collection: {collection_name}")

    for file in tqdm(os.listdir(json_dirpath)):
        path = os.path.join(json_dirpath, file)
        with open(path) as f:
            data = json.load(f)

        # if we work with an older version of exported data (containing 'doc_id' field), we need
        # to firstly convert it to 'asset_field'
        if len(data) > 0 and "doc_id" in data[0]:
            for i in range(len(data)):
                data[i]["asset_id"] = int(data[i].pop("doc_id"))

        data = [d for d in data if d["asset_id"] not in unique_asset_ids]
        if len(data) == 0:
            continue

        for i in range(len(data)):
            data[i]["vector"] = np.array(data[i]["vector"])

            # remove irrelevant fields from the data pertaining to metadata
            if extract_metadata is False:
                fields_to_del = set(data[i].keys()) - {"asset_id", "vector"}
                for field in fields_to_del:
                    data[i].pop(field)

        client.insert(collection_name=collection_name, data=data)
        unique_asset_ids.update(d["asset_id"] for d in data)


def create_new_collection(
    client: MilvusClient, collection_name: str, extract_metadata: bool
) -> bool:
    if client.has_collection(collection_name):
        return False

    vector_index_kwargs = {
        "index_type": "HNSW_SQ",
        "metric_type": "COSINE",
        "params": {"sq_type": "SQ8"},
    }
    scalar_index_kwargs = {"index_type": "INVERTED"}

    schema = client.create_schema(auto_id=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field("asset_id", DataType.INT64)

    if extract_metadata and collection_name.endswith("_datasets"):
        # TODO for now this is a duplicate code to the index creation process found
        # in the embedding_store.py file

        # TODO so far it only works with datasets
        # (the same goes for the Milvus index in the main application)
        schema.add_field("date_published", DataType.VARCHAR, max_length=22, nullable=True)
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

    index_params = client.prepare_index_params()

    index_params.add_index(field_name="vector", **vector_index_kwargs)
    index_params.add_index(field_name="asset_id", **scalar_index_kwargs)

    # TODO This has been intentionally commented out due to unexpected Milvus behavior...
    # Issue: https://github.com/aiondemand/aiod-enhanced-interaction/issues/77
    # if extract_metadata and collection_name.endswith("_datasets"):
    #     index_params.add_index(field_name="date_published", **scalar_index_kwargs)
    #     index_params.add_index(field_name="size_in_mb", **scalar_index_kwargs)
    #     index_params.add_index(field_name="license", **scalar_index_kwargs)
    #     index_params.add_index(field_name="task_types", **scalar_index_kwargs)
    #     index_params.add_index(field_name="languages", **scalar_index_kwargs)
    #     index_params.add_index(field_name="datapoints_upper_bound", **scalar_index_kwargs)
    #     index_params.add_index(field_name="datapoints_lower_bound", **scalar_index_kwargs)

    client.create_collection(
        collection_name=collection_name, schema=schema, index_params=index_params
    )
    return True


if __name__ == "__main__":
    parser = ArgumentParser(description="Populate database with embeddings from JSON files.")
    parser.add_argument(
        "-i",
        "--input_dirpath",
        type=str,
        required=True,
        help="Path to the directory containing JSON files for database population",
    )
    parser.add_argument(
        "--uri",
        type=str,
        required=False,
        help="URI of the local Milvus database",
        default="http://localhost:19530",
    )
    parser.add_argument(
        "-u",
        "--username",
        type=str,
        required=False,
        help="Username of the Milvus user to connect to the local Milvus database",
        default="root",
    )
    parser.add_argument(
        "-p",
        "--password",
        type=str,
        required=False,
        help="Password of the Milvus user to connect to the local Milvus database",
        default="Milvus",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=False,
        help="Whether we wish to create an Milvus indices accounting for metadata to extract",
        default="false",
    )

    try:
        args = InputArgs(**parser.parse_args().__dict__)
        populate_database(args)
        exit(0)
    except Exception as e:
        print(e)
        exit(1)
