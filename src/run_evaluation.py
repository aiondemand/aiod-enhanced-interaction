import sys
import os
from langchain_community.llms import Ollama

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(src_dir)

from evaluation.llm import load_llm
from model.base import EmbeddingModel, RetrievalSystem
from preprocess.text_operations import ConvertJsonToString
from embedding_stores import Chroma_EmbeddingStore, EmbeddingStore
from evaluation.hit_rate_evaluation import HitRateEvaluationPipeline
from evaluation.llm_evaluator import LLM_Evaluator
from evaluation.precision_evaluation import PrecisionEvaluationPipeline
from model.embedding_models.setup import ModelSetup
from utils import init
from model.retrieval import EmbeddingModel_Pipeline, RAG_Pipeline


def precision_evaluation(
    model_names: list[str], process_text_types: list[str],    
    heuristic_function: bool = False, topk: int = 10, chunk_embeddings: bool = False
) -> None:
    client = init()
    store = Chroma_EmbeddingStore(
        client, chunk_embedding_store=chunk_embeddings, verbose=True
    )

    score_function, relevance_function = None, None
    if heuristic_function:
        score_function = LLM_Evaluator.heuristic_score_function
        relevance_function = lambda x: x >= 0.65

    for model_name in model_names:
        for process_text_type in process_text_types:
            asset_text_dirpath = f"./data/{process_text_type}-texts"        
            folder_model_name = (
                f"{model_name}--CHUNK_EMBEDS--{process_text_type}"
                if chunk_embeddings
                else f"{model_name}--{process_text_type}"
            )

            retrieval_system = load_retrieval_system(
                model_name, store, topk, process_text_type, 
                chunk_embeddings=chunk_embeddings
            )
            pipeline = PrecisionEvaluationPipeline(
                retrieval_system,
                model_name=folder_model_name,
                asset_type="dataset",
                generate_queries_dirpath="./data/queries/generic",
                asset_text_dirpath=asset_text_dirpath,
                topk_dirpath=f"./data/topk-results/topk-results-{topk}",
                llm_eval_dirpath="./data/llm_evaluations",
                annotated_query_dirpath=f"./data/annotated-queries",
                metrics_dirpath=f"./data/results/precision",
            )
            pipeline.execute(score_function, relevance_function)


def recall_evaluation(
    model_names: list[str], process_text_types: list[str],
    topk: int = 30, chunk_embeddings: bool = False
) -> None:
    client = init()
    store = Chroma_EmbeddingStore(
        client, chunk_embedding_store=chunk_embeddings, verbose=True
    )

    for model_name in model_names:
        for process_text_type in process_text_types:
            folder_model_name = (
                f"{model_name}--CHUNK_EMBEDS--{process_text_type}"
                if chunk_embeddings
                else f"{model_name}--{process_text_type}"
            )

            retrieval_system = load_retrieval_system(
                model_name, store, topk, process_text_type, 
                chunk_embeddings=chunk_embeddings
            )
            pipeline = HitRateEvaluationPipeline(
                retrieval_system,
                model_name=folder_model_name,
                orig_json_assets_dirpath="./data/jsons",
                quality_assets_path="./data/queries/asset-specific/handpicked_datasets.json",
                generate_queries_dirpath="./data/queries/asset-specific",
                topk_dirpath=f"./data/topk-results/topk-results-{topk}",
                metrics_dirpath=f"./data/results/hit_rate",
            )
            pipeline.execute()
    

# TODO we should move this somewhere else...
def load_retrieval_system(
    model_name: str, embedding_store: EmbeddingStore, topk: int,
    process_text_type: str, chunk_embeddings: bool = False
) -> RetrievalSystem:
    if model_name.startswith("RAG-"):
        _, llm_name, embedding_model_name = model_name.split("-")

        llm = load_llm(ollama_name=llm_name if llm_name != "gpt_4o" else None)
        embedding_model = load_embedding_model(embedding_model_name) 
        collection_name = (
            f"chunk_embeddings-{embedding_model_name}-{process_text_type}-v0"
            if chunk_embeddings
            else f"embeddings-{embedding_model_name}-{process_text_type}-v0"
        )

        return RAG_Pipeline(
            embedding_model, 
            embedding_store,
            emb_collection_name=collection_name,
            document_collection_name="datasets",
            stringify_document_func=ConvertJsonToString.extract_very_basic_info,
            retrieval_topk=topk*5,
            output_topk=topk,
            llm=llm
        )

    embedding_model = load_embedding_model(model_name)
    collection_name = (
        f"chunk_embeddings-{model_name}-{process_text_type}-v0"
        if chunk_embeddings
        else f"embeddings-{model_name}-{process_text_type}-v0"
    )

    return EmbeddingModel_Pipeline(
        embedding_model, embedding_store, topk=topk, 
        emb_collection_name=collection_name
    )
    
        
def load_embedding_model(model_name: str) -> EmbeddingModel:
    if model_name == "gte_large":
        return ModelSetup._setup_gte_large()
    if model_name == "gte_large_hierarchical":
        return ModelSetup._setup_gte_large_hierarchical()
    if model_name == "multilingual_e5_large":
        return ModelSetup._setup_multilingual_e5_large()
    if model_name == "bge_large":
        return ModelSetup._setup_bge_large()
    
    raise ValueError("Unsupported model for evaluation")


if __name__ == "__main__":
    process_text_types = ["basic", "relevant"]

    ##### DOC EMBEDDINGS #####
    model_names = ["gte_large", "multilingual_e5_large"]
    chunk_embeddings = False

    # precision_evaluation(
    #     model_names, process_text_types, topk=10, chunk_embeddings=chunk_embeddings,
    #     heuristic_function=False
    # )
    # precision_evaluation(
    #     model_names, process_text_types, topk=10, chunk_embeddings=chunk_embeddings,
    #     heuristic_function=True
    # )
    recall_evaluation(model_names, process_text_types, topk=30, chunk_embeddings=chunk_embeddings)


    ##### CHUNK EMBEDDINGS #####
    model_names = ["multilingual_e5_large", "bge_large"]
    chunk_embeddings = True

    # precision_evaluation(
    #     model_names, process_text_types, topk=10, chunk_embeddings=chunk_embeddings,
    #     heuristic_function=False
    # )
    # precision_evaluation(
    #     model_names, process_text_types, topk=10, chunk_embeddings=chunk_embeddings,
    #     heuristic_function=True
    # )
    recall_evaluation(model_names, process_text_types, topk=30, chunk_embeddings=chunk_embeddings)