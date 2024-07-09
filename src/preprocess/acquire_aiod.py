import os
import json
import requests
from tqdm import tqdm
from time import sleep
from tqdm import tqdm
import json
import os
from chromadb.api.client import Client

import utils


def populate_database_with_assets(base_url: str, asset_name: str, savedir: str) -> None:
    new_collection_name = f"{asset_name}-docs"    
    chroma_client = utils.init(return_chroma_client=True)
    collection_names = [col.name for col in chroma_client.list_collections()]

    if new_collection_name in collection_names:
        raise ValueError(f"Collection '{new_collection_name}' already exists")

    get_aiod_assets(base_url, asset_name, savedir)
    create_document_collection(chroma_client, new_collection_name, savedir)


def get_aiod_assets(
    base_url: str, asset_name: str, savedir: str, win_size: int = 1_000,
    starting_offset: int = 0, sleep_duration: int = 3 
) -> None:
    total_number = _perform_request(f"{base_url}/counts/{asset_name}/v1")
    url = f"{base_url}/{asset_name}/v1"
    os.makedirs(savedir, exist_ok=True)

    for offset in tqdm(
        range(starting_offset, total_number, win_size), 
        total=(total_number - starting_offset) // win_size
    ):
        queries = {
            "schema": "aiod",
            "offset": offset,
            "limit": win_size
        }  
        datasets = _perform_request(url, queries)

        if (
            (
                # all but the last request
                offset + win_size <= total_number and
                len(datasets) != win_size 
            ) or
            (   
                # last get request 
                offset + win_size > total_number and 
                len(datasets) != total_number % win_size
            )
        ):
            raise ValueError("Not all assets have been correctly downloaded")

        with open(os.path.join(savedir, f"{asset_name}_{offset}.json"), "w") as f:
            json.dump(datasets, f, ensure_ascii=False)
        sleep(sleep_duration)


def _perform_request(
    url: str, params: dict | None = None, num_retries: int = 3, 
    timeout_sleep_duration: int = 60
) -> dict | None:
    for _ in range(num_retries):
        try:
            return requests.get(url, params, timeout=60).json()
        except requests.exceptions.ConnectTimeout:
            sleep(timeout_sleep_duration)
    
    raise ValueError("We couldn't connect to AIoD API")
    

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
    # base_url = "https://aiod-dev.i3a.es"
    base_url = "https://api.aiod.eu"

    populate_database_with_assets(base_url, "datasets", "./temp/datasets")
