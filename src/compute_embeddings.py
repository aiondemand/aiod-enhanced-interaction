from chromadb import Client
from dataset import AIoD_Documents
from embedding_stores import Chroma_EmbeddingStore
from model.base import EmbeddingModel
from model.embedding_models.setup import ModelSetup
import utils


def store_embeddings_wrapper():
    client = utils.init()
    collection_name_placeholder = "chunk_embeddings-{model_name}-{text_type}-v0"
    process_text_types = ["relevant", "basic"]
    model_names = ["multilingual_e5_large", "bge_large"]
    loader_kwargs = {
        "num_workers": 1
    }

    for model_name in model_names:
        if model_name == "gte_large":
            embedding_model = ModelSetup._setup_gte_large()
            loader_kwargs["batch_size"] = 8
        elif model_name == "multilingual_e5_large":
            embedding_model = ModelSetup._setup_multilingual_e5_large()
            loader_kwargs["batch_size"] = 32
        elif model_name == "bge_large":
            embedding_model = ModelSetup._setup_bge_large()
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
                loader_kwargs=loader_kwargs
            )


def store_embeddings(
        model: EmbeddingModel, client: Client, text_dirpath: str, 
        collection_name: str, loader_kwargs: dict | None = None
    ) -> None:
    ds = AIoD_Documents(text_dirpath, testing_random_texts=False)
    ds.filter_out_already_computed_docs(client, collection_name)
    loader = ds.build_loader(loader_kwargs)

    store = Chroma_EmbeddingStore(client, verbose=True)
    store.store_embeddings(model, loader, collection_name)


if __name__ == "__main__":
    store_embeddings_wrapper()
