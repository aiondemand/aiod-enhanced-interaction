from typing import Literal
import torch
import numpy as np
import os
from dotenv import load_dotenv
import sys
import chromadb
from chromadb.api.client import Client as ChromaClient
from chromadb.config import Settings
from pymilvus import MilvusClient

from data_types import VectorDbClient

CHROMA_CLIENT_AUTH_PROVIDER = "chromadb.auth.token_authn.TokenAuthClientProvider"


class HideOutput:
    def __enter__(self):
        self.stderr = sys.stderr
        self.stdout = sys.stdout
        self.devnull = open(os.devnull, 'w')
        sys.stdout = self.devnull
        sys.stderr = self.devnull
        
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stderr = self.stderr
        sys.stdout = self.stdout
        self.devnull.close()


def connect_to_chroma() -> ChromaClient:    
    try:
        client = chromadb.HttpClient(
            host=os.environ.get("CHROMA_HOST"), 
            port=os.environ.get("CHROMA_PORT"),
            settings=Settings(
                chroma_client_auth_provider=CHROMA_CLIENT_AUTH_PROVIDER,
                chroma_client_auth_credentials=os.environ.get("CHROMA_TOKEN")
            )
        )
    except Exception as e:
        print("Failed to connect to the ChromaDB.")
        raise e
    
    return client
    

def connect_to_milvus() -> MilvusClient:
    try:
        milvus_token = f"{os.environ.get('MILVUS_USER')}:{os.environ.get('MILVUS_PASS')}"
        client = MilvusClient(
            uri=os.environ.get("MILVUS_URI"),
            token=milvus_token
        ) 
    except Exception as e:
        print("Failed to connect to the MilvusDB.")
        raise e
    
    return client


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def init(
    return_db_client: bool = True, 
    vector_db_to_use: Literal["chroma", "milvus", "env"] = "env"
) -> None | VectorDbClient:
    """
    Initialize all the env vars and randomness mechanisms for reproducibility reasons
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false" 
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_float32_matmul_precision('medium')
    no_randomness(seed=0)
    load_dotenv()

    if return_db_client is False:
        return None
    if vector_db_to_use == "env":
        vector_db_to_use = os.environ.get("VECTOR_DB_TO_USE", "milvus")

    if vector_db_to_use == "chroma":
        return connect_to_chroma()
    if vector_db_to_use == "milvus":
        return connect_to_milvus()


def no_randomness(seed: int = 0) -> None:
    np.random.seed(seed)
    np.random.default_rng(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
