from pymilvus import MilvusClient
from pymilvus import DataType
from torch.utils.data import DataLoader
from chromadb.api.client import Client
from tqdm import tqdm
import torch
import uuid
import numpy as np

from model.lm_encoders.setup import ModelSetup
from model.lm_encoders.models import RepresentationModel
import utils
from dataset import AIoD_Documents


class Chroma_EmbeddingStore:
    def __init__(self, client: Client, verbose: bool = False) -> None:
        self.client = client
        self.verbose = verbose

    def _get_collection(self, collection_name: str):
        try:
            collection = self.client.get_collection(collection_name)
        except Exception as e:
            print(f"Collection '{collection_name}' doesn't exist.")
            raise e
        
        return collection

    def create_collection(self, collection_name: str) -> None:
        self.client.create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine"
            },
            get_or_create=True
        )

    def store_embeddings(
        self, model: RepresentationModel, loader: DataLoader, collection_name: str,
        chroma_batch_size: int = 10
    ) -> None:
        was_training = model.training
        model.eval()
        collection = self._get_collection(collection_name)

        all_embeddings = []
        all_ids = []
        all_meta = []
        for it, (texts, doc_ids) in tqdm(
            enumerate(loader), total=len(loader), disable=self.verbose is False
        ):
            with torch.no_grad():
                embeddings = model(texts)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_ids.extend([str(uuid.uuid4()) for _ in range(len(doc_ids))])
            all_meta.extend([{"doc_id": id} for id in doc_ids])
    
            if len(all_embeddings) == chroma_batch_size or it == len(loader) - 1:
                all_embeddings = np.vstack(all_embeddings)
                collection.add(
                    embeddings=embeddings, 
                    ids=all_ids,
                    metadatas=all_meta
                )

                all_embeddings = []
                all_ids = []
                all_meta = []

        if was_training:
            model.train()

    def semantic_search(
        self, model: RepresentationModel, query_list: list[tuple[str, int]],
        collection_name: str, topk: int = 10, chroma_batch_size: int = 10
    ) -> list:
        was_training = model.training
        model.eval()
        collection = self._get_collection(collection_name)

        all_results = []
        all_embeddings = []        
        query_loader = DataLoader(
            query_list, collate_fn=AIoD_Documents._collate_fn, batch_size=4, num_workers=2
        )
        for it, (texts, _) in tqdm(
            enumerate(query_loader), total=len(query_loader), disable=self.verbose is False
        ):
            with torch.no_grad():
                embeddings = model(texts)
            all_embeddings.append(embeddings.cpu().numpy())
            
            if len(all_embeddings) == chroma_batch_size or it == len(loader) - 1:
                all_embeddings = np.vstack(all_embeddings)

                sem_search_results = collection.query(
                    query_embeddings=embeddings,
                    n_results=topk,
                    include=["metadatas", "distances"]
                )
                # TODO postprocess the retrieved results
                all_results.extend(sem_search_results)
                all_embeddings = []
        
        if was_training:
            model.train()
        return all_results


if __name__ == "__main__":
    chroma_client = utils.init()
    model = ModelSetup.setup_model_wrapper(
        "sentence-transformers/all-MiniLM-L12-v2", 
        pooling="mean", is_hierarchical=True, max_num_chunks=2
    )

    text_dirpath = "./data/extracted_data"
    ds = AIoD_Documents(text_dirpath)
    loader = ds.build_loader({"batch_size": 32, "num_workers": 4})

    collection_name = "test_collection"    
    store = Chroma_EmbeddingStore(chroma_client, verbose=True)
    
    # COMPUTE EMBEDDINGS
    store.create_collection(collection_name)
    store.store_embeddings(model, loader, collection_name)

    # PERFORM SEMANTIC SEARCH
    query_list = [
        ("Question 1", "1"),
        ("Question 2", "2"),
    ]
    store.semantic_search(model, query_list, collection_name, topk=10)