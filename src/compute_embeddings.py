import sys
import os
from chromadb.api.client import Client

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(src_dir)

from dataset import AIoD_Documents
from embedding_stores import Chroma_EmbeddingStore
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
                loader_kwargs=loader_kwargs
            )


def store_embeddings(
        model: EmbeddingModel, client: Client, text_dirpath: str, 
        collection_name: str, chunk_embeddings: bool = False,
        loader_kwargs: dict | None = None
    ) -> None:
    ds = AIoD_Documents(text_dirpath, testing_random_texts=False)
    ds.filter_out_already_computed_docs(client, collection_name)
    loader = ds.build_loader(loader_kwargs)

    store = Chroma_EmbeddingStore(client, chunk_embedding_store=chunk_embeddings, verbose=True)
    store.store_embeddings(model, loader, collection_name)


if __name__ == "__main__":
    process_text_types = ["basic", "relevant"] 
    model_names = ["gte_large_hierarchical"]

    store_embeddings_wrapper(model_names, process_text_types, chunk_embeddings=True)
