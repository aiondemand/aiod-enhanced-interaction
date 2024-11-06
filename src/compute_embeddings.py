import sys
import os
from chromadb.api.client import Client as ChromaClient
from pymilvus import MilvusClient

from data_types import VectorDbClient
from dataset import AIoD_Documents
from embedding_stores import Chroma_EmbeddingStore, Milvus_EmbeddingStore
from model.base import EmbeddingModel
from model.embedding_models.setup import ModelSetup
import utils


def store_embeddings_wrapper(
    model_names: list[str], process_text_types: list[str], 
    chunk_embeddings: bool = False
) -> None:
    client = utils.init()
    collection_name_placeholder = (
        "chunk_embeddings-{model_name}-{text_type}-v0"
        if chunk_embeddings
        else "embeddings-{model_name}-{text_type}-v0"
    )
    if type(client) == MilvusClient:
        collection_name_placeholder = (
            "chunk_embeddings_{model_name}_{text_type}"
            if chunk_embeddings
            else "embeddings_{model_name}_{text_type}"
        )   

    loader_kwargs = {
        "num_workers": 1
    }

    for model_name in model_names:
        if model_name == "gte_large":
            embedding_model = ModelSetup._setup_gte_large()
            loader_kwargs["batch_size"] = 8
        if model_name == "gte_large_hierarchical":
            embedding_model = ModelSetup._setup_gte_large_hierarchical()
            loader_kwargs["batch_size"] = 16
        elif model_name == "multilingual_e5_large":
            embedding_model = ModelSetup._setup_multilingual_e5_large()
            loader_kwargs["batch_size"] = 16
        elif model_name == "bge_large":
            embedding_model = ModelSetup._setup_bge_large()
            loader_kwargs["batch_size"] = 32
        else:
            raise ValueError("Unsupported model for evaluation")
        
        # TODO I reduced the computational requirements...
        loader_kwargs["num_workers"] = 0
        loader_kwargs["batch_size"] = 2

        for process_text_type in process_text_types:
            text_dirpath = f"data/{process_text_type}-texts"
            collection_name = collection_name_placeholder.format(
              model_name = model_name,
              text_type=process_text_type
            )
            store_embeddings(
                embedding_model, 
                client,
                text_dirpath=text_dirpath, 
                collection_name=collection_name,
                chunk_embeddings=chunk_embeddings,
                loader_kwargs=loader_kwargs,
                emb_dimensionality=embedding_model.input_transformer.embedding_dim
            )


def store_embeddings(
    model: EmbeddingModel, client: VectorDbClient, text_dirpath: str, 
    collection_name: str, chunk_embeddings: bool = False,
    loader_kwargs: dict | None = None, emb_dimensionality: int | None = None
) -> None:
    ds = AIoD_Documents(text_dirpath, testing_random_texts=False)
    ds.filter_out_already_computed_docs(client, collection_name)
    loader = ds.build_loader(loader_kwargs)

    if type(client) == ChromaClient:
        store = Chroma_EmbeddingStore(client, chunk_embedding_store=chunk_embeddings, verbose=True)
    elif type(client) == MilvusClient:
        store = Milvus_EmbeddingStore(
            client, emb_dimensionality=emb_dimensionality, 
            chunk_embedding_store=chunk_embeddings, verbose=True
        )
    else:
        raise ValueError("Invalid DB client")

    # TODO
    # store.store_embeddings(model, loader, collection_name)

    # TODO get rid of the remaining code later on
    queries = [
        {
            "text": "Searching for datasets about stock market",
            "id": "id1"
        },
        {
            "text": "Animal datasets",
            "id": "id2"
        }
    ]
    from dataset import Queries
    query_loader = Queries(queries=queries).build_loader()

    topk_documents = store.retrieve_topk_document_ids(model, query_loader, topk=10, emb_collection_name=collection_name)
    topk_documents


if __name__ == "__main__":
    process_text_types = ["relevant"] 
    model_names = ["gte_large_hierarchical"]

    store_embeddings_wrapper(model_names, process_text_types, chunk_embeddings=True)
