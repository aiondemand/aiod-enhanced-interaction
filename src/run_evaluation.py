import sys
import os

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(src_dir)

from embedding_stores import Chroma_EmbeddingStore
from evaluation.hit_rate_evaluation import HitRateEvaluationPipeline
from evaluation.precision_evaluation import PrecisionEvaluationPipeline
from model.embedding_models.setup import ModelSetup
from utils import init
from model.retrieval import EmbeddingModel_Pipeline


def precision_evaluation(topk: int = 10) -> None:
    client = init()
    store = Chroma_EmbeddingStore(client, verbose=True)

    model_names = ["gte_large", "multilingual_e5_large"]
    process_text_types = ["basic", "relevant"]

    for model_name in model_names:
        if model_name == "gte_large":
            embedding_model = ModelSetup._setup_gte_large(model_max_length=4096)
        elif model_name == "multilingual_e5_large":
            embedding_model = ModelSetup._setup_multilingual_e5_large()
        else:
            raise ValueError("Unsupported model for evaluation")
        
        for process_text_type in process_text_types:
            asset_text_dirpath = f"./data/{process_text_type}-texts"
            folder_model_name = f"{model_name}--{process_text_type}"
            collection_name = f"embeddings-{model_name}-{process_text_type}-v0"

            retrieval_system = EmbeddingModel_Pipeline(
                embedding_model, store, topk=topk, 
                emb_collection_name=collection_name
            )
            pipeline = PrecisionEvaluationPipeline(
                retrieval_system,
                model_name=folder_model_name,
                asset_type="dataset",
                generate_queries_dirpath="./data/queries/generic",
                asset_text_dirpath=asset_text_dirpath,
                topk_dirpath="./data/topk-results",
                llm_eval_dirpath="./data/llm_evaluations",
                annotated_query_dirpath=f"./data/annotated-queries",
                metrics_dirpath=f"./data/results/precision",
                post_process_llm_prediction_function = None,
            )
            pipeline.execute()


def recall_evaluation(topk: int = 100) -> None:
    client = init()
    store = Chroma_EmbeddingStore(client, verbose=True)

    model_names = ["gte_large", "multilingual_e5_large"]
    process_text_types = ["basic", "relevant"]

    for model_name in model_names:
        if model_name == "gte_large":
            embedding_model = ModelSetup._setup_gte_large(model_max_length=4096)
        elif model_name == "multilingual_e5_large":
            embedding_model = ModelSetup._setup_multilingual_e5_large()
        else:
            raise ValueError("Unsupported model for evaluation")
        
        for process_text_type in process_text_types:
            folder_model_name = f"{model_name}--{process_text_type}"
            collection_name = f"embeddings-{model_name}-{process_text_type}-v0"

            retrieval_system = EmbeddingModel_Pipeline(
                embedding_model, store, topk=topk, 
                emb_collection_name=collection_name
            )
            pipeline = HitRateEvaluationPipeline(
                retrieval_system,
                model_name=folder_model_name,
                orig_json_assets_dirpath="./data/jsons",
                quality_assets_path="./data/queries/asset-specific/handpicked_datasets.json",
                generate_queries_dirpath="./data/queries/asset-specific",
                topk_dirpath="./data/topk-results",
                metrics_dirpath=f"./data/results/hit_rate",
            )
            pipeline.execute()
    

if __name__ == "__main__":
    # precision_evaluation()
    recall_evaluation()
