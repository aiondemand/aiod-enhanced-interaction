import json
import requests
from tqdm import tqdm
from time import sleep

"""
This file was used for retrieving documents from AIoD API
"""


BASE_URL = "https://aiod-dev.i3a.es"

if __name__ == "__main__":
    total_number = requests.get(f"{BASE_URL}/counts/datasets/v1").json()
    starting_offset = 0
    win_size = 1_000
    url = f"{BASE_URL}/datasets/v1"
    
    for offset in tqdm(
        range(starting_offset, total_number, win_size), 
        total=(total_number - starting_offset) // win_size
    ):
        queries = {
            "schema": "aiod",
            "offset": offset,
            "limit": win_size
        }  
        try:
            response = requests.get(url, queries)
            datasets = response.json()
        except requests.exceptions.ConnectTimeout:
            print("ConnectTimeout exception occurred. Let's wait for a minute and then try again")
            sleep(60)
            response = requests.get(url, queries)
            datasets = response.json()


        with open(f"./data/jsons/datasets_{offset}.json", "w") as f:
            json.dump(datasets, f, ensure_ascii=False)

        sleep(3)