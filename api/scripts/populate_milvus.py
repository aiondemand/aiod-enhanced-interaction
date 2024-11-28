import json
import os
from argparse import ArgumentParser
from time import sleep

import numpy as np
from pydantic import BaseModel
from pymilvus import DataType, MilvusClient
from pymilvus.milvus_client import IndexParams
from tqdm import tqdm


class InputArgs(BaseModel):
    input_dirpath: str
    uri: str
    username: str
    password: str


def populate_database(args: InputArgs) -> None:
    if os.path.exists(args.input_dirpath) is False:
        exit(1)

    sleep(10)  # Headstart for Milvus to fully initialize
    client = MilvusClient(uri=args.uri, user=args.username, password=args.password)
    for collection_name in sorted(os.listdir(args.input_dirpath)):
        path = os.path.join(args.input_dirpath, collection_name)
        populate_collection(client, collection_name, path)


def populate_collection(
    client: MilvusClient, collection_name: str, json_dirpath: str
) -> None:
    created_new = create_new_collection(client, collection_name)
    if created_new is False:
        return

    print(f"Populating collection: {collection_name}")
    for file in tqdm(os.listdir(json_dirpath)):
        path = os.path.join(json_dirpath, file)
        with open(path) as f:
            data = json.load(f)
        for i in range(len(data)):
            data[i]["vector"] = np.array(data[i]["vector"])

        client.insert(collection_name=collection_name, data=data)


def create_new_collection(client: MilvusClient, collection_name: str) -> bool:
    if client.has_collection(collection_name):
        return False

    schema = client.create_schema(auto_id=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=20)
    schema.verify()

    index_params = IndexParams()
    index_params.add_index("vector", "", "", metric_type="COSINE")
    index_params.add_index("doc_id", "", "")

    client.create_collection(
        collection_name=collection_name,
        dimension=1024,
        auto_id=True,
    )
    return True


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Populate database with embeddings from JSON files."
    )
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

    try:
        args = InputArgs(**parser.parse_args().__dict__)
        populate_database(args)
        exit(0)
    except Exception as e:
        print(e)
        exit(1)
