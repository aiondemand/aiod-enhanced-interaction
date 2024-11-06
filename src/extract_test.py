from tqdm import tqdm
import json
import sys
import os

from data_types import VectorDbClient
from dataset import AIoD_Documents
from embedding_stores import Chroma_EmbeddingStore, Milvus_EmbeddingStore
from lang_chains import load_llm
from model.base import EmbeddingModel
from model.embedding_models.setup import ModelSetup
import utils
from llm_metadata_filter import LLM_MetadataExtractor
from preprocess.text_operations import ConvertJsonToString

def save_text_data(json_dirpath: str, savedir: str) -> None:
    os.makedirs(savedir, exist_ok=True)
    with open(json_dirpath) as f:
        json_data = json.load(f)
    
    extracted_texts = [
        ConvertJsonToString.extract_relevant_info(x, stringify=False) for x in json_data
    ]

    for orig_json, text in tqdm(zip(json_data, extracted_texts), total=len(json_data)):
        id = orig_json["identifier"]
        with open(os.path.join(savedir, f"{id}.txt"), "w") as f:
            f.write(text)


if __name__ == "__main__":        
    client = utils.init()

    json_dirpath = "./temp/data_examples/huggingface.json"
    text_dirpath = "./data/extract-sample-data"
    collection_name = "extract_metadata_demo"
    # save_text_data(json_dirpath=json_dirpath, savedir=text_dirpath)

    embedding_model = ModelSetup._setup_gte_large_hierarchical()

    ds = AIoD_Documents(text_dirpath, testing_random_texts=False)
    ds.filter_out_already_computed_docs(client, collection_name)
    loader = ds.build_loader(loader_kwargs={"batch_size": 2, "num_workers": 0})

    store = Milvus_EmbeddingStore(
        client, emb_dimensionality=1024, 
        chunk_embedding_store=True, 
        extract_metadata=True,
        verbose=True
    )

    llm_chain = LLM_MetadataExtractor.build_chain(llm=load_llm(), parsing_user_query=False)
    llm_extractor = LLM_MetadataExtractor(
        chain=llm_chain,
        asset_type="dataset", 
        parsing_user_query=False,
    )

    # store.store_embeddings(
    #     embedding_model, loader, collection_name, milvus_batch_size=5,
    #     extract_metadata_llm=llm_extractor
    # )

    queries = [
        {
            "text": "Multilingual dataset containing English and French data",
            "id": "id1"
        },
        {
            "text": "Tabular dataset with more than 10 000 datapoints",
            "id": "id2"
        }
    ]
    from dataset import Queries
    query_loader = Queries(queries=queries).build_loader(loader_kwargs={"batch_size": 2, "num_workers": 0})

    llm_chain_user_query = LLM_MetadataExtractor.build_chain(llm=load_llm(), parsing_user_query=True)
    extract_conditions_llm = LLM_MetadataExtractor(
        chain=llm_chain_user_query,
        asset_type="dataset", 
        parsing_user_query=True,
    )

    store.retrieve_topk_document_ids(
        embedding_model, query_loader, emb_collection_name=collection_name,
        extract_conditions_llm=extract_conditions_llm
    )
    