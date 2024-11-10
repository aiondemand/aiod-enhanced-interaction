from argparse import ArgumentParser
import os
import json
import numpy as np

from pydantic import BaseModel
from pymilvus import MilvusClient, DataType
from pymilvus.milvus_client import IndexParams
from tqdm import tqdm


class InputArgs(BaseModel):
    input_dirpath: str
    username: str
    password: str


def populate_database(args: InputArgs) -> None:
    client = MilvusClient(
        token=f"{args.username}:{args.password}"
    )
    for collection_name in os.listdir(args.input_dirpath):
        path = os.path.join(args.input_dirpath, collection_name)
        populate_collection(client, collection_name, path)
    

def populate_collection(
    client: MilvusClient, collection_name: str, json_dirpath: str
) -> None:
    create_collection(client, collection_name)
    print(f"Populating collection: {collection_name}")
    
    for file in tqdm(os.listdir(json_dirpath)):
        path = os.path.join(json_dirpath, file)
        with open(path) as f:
            data = json.load(f)
        for i in range(len(data)):
            data[i]["vector"] = np.array(data[i]["vector"])
    
        client.insert(collection_name=collection_name, data=data)
    

def create_collection(client: MilvusClient, collection_name: str) -> None:
    if client.has_collection(collection_name):
        return 
    
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


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Populate database with embeddings from JSON files."
    ) 
    parser.add_argument(
        '-i', '--input_dirpath',
        type=str,
        required=True,
        help='Path to the directory containing JSON files for database population'
    )
    parser.add_argument(
        '-u', '--username',
        type=str,
        required=True,
        help='Username of the Milvus user to connect to the local Milvus database'
    )
    parser.add_argument(
        '-p', '--password',
        type=str,
        required=True,
        help='Password of the Milvus user to connect to the local Milvus database'
    )

    args = InputArgs(**parser.parse_args().__dict__)
    populate_database(args)
    