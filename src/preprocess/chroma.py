from tqdm import tqdm
import json
import os
from chromadb.api.client import Client

import utils

"""
This file was used for populating chromadb with documents retrieved from AIoD API
"""


def dummy_embeddings(texts: str) -> list[list[float]]:
    return [[0] for _ in range(len(texts))]


def create_document_collection(
    client: Client, collection_name: str, json_dirpath: str,
    chroma_batch_size: int = 10
) -> None:
    try:
        client.delete_collection(name=collection_name)
    except:
        pass

    collection = client.create_collection(
        name=collection_name
    )
    filenames = sorted(os.listdir(json_dirpath))

    counter = 0
    all_meta = []
    all_ids = []
    for it, file in tqdm(
        enumerate(filenames), total=len(filenames)
    ):
        json_filepath = os.path.join(json_dirpath, file)
        with open(json_filepath) as f:
            data = json.load(f)

        counter += 1
        all_ids.extend([str(obj["identifier"]) for obj in data])
        all_meta.extend([{"json_string": json.dumps(obj)} for obj in data])

        if counter == chroma_batch_size or it == len(filenames) - 1:
            collection.add(
                embeddings=dummy_embeddings(all_ids),
                metadatas=all_meta,
                ids=all_ids
            )

            all_meta = []
            all_ids = []
            counter = 0


if __name__ == "__main__":
    client = utils.init(return_chroma_client=True)
    json_dirpath = "./data/jsons"
    create_document_collection(
        client, collection_name="datasets", 
        json_dirpath=json_dirpath
    )