from typing import Callable
import numpy as np
import os
import torch

from dataset import Queries
from embedding_stores import EmbeddingStore
from evaluation.llm import LLM_Evaluator, build_query_json_from_llm_eval, evaluate_query_doc_pairs
from evaluation.metrics import RetrievalEvaluation
from evaluation.query_generation.base import QueryGeneration
from lang_chains import Chain
from model.base import EmbeddingModel


class PrecisionEvaluationPipeline:
    def __init__(
        self, emb_model: EmbeddingModel, 
        embedding_store: EmbeddingStore,
        query_generation: QueryGeneration, 
        llm: Chain, 
        asset_text_dirpath: str, 
        topk_dirpath: str, 
        llm_eval_dirpath: str,
        query_jsonfile_path: str,
        metrics_savepath: str,
        post_process_llm_prediction_function: Callable[[dict], dict] | None = None,
        retrieve_topk_documents_func_kwargs: dict | None = None,
        verbose: bool = True
    ) -> None:
        self.emb_model = emb_model
        self.embedding_store = embedding_store
        self.query_generation = query_generation
        self.llm = llm
        
        self.asset_text_dirpath = asset_text_dirpath
        self.topk_dirpath = topk_dirpath
        self.llm_eval_dirpath = llm_eval_dirpath
        self.query_jsonfile_path = query_jsonfile_path
        self.metrics_savepath = metrics_savepath
        
        self.post_process_llm_prediction_function = post_process_llm_prediction_function
        if post_process_llm_prediction_function is None:
            self.post_process_llm_prediction_function = lambda x: x
        self.retrieve_topk_documents_func_kwargs = retrieve_topk_documents_func_kwargs
        if retrieve_topk_documents_func_kwargs is None:
            self.retrieve_topk_documents_func_kwargs = {}
        self.verbose = verbose

    def execute(
        self, 
        topk: int = 10, 
        score_function: Callable[[dict], float] | None = None,
        relevance_function: Callable[[float], bool] | None = None
    ) -> None:
        # TODO perform query generation
        print("=========== PRECISION EVALUATION STARTED ===========")

        print("=== Generation of generic queries ===")
        # QUERY GENERATION PART SKIPPED FOR NOW

        queries = [
            {
                "text": "Looking for a dataset with labeled text data in English for sentiment analysis, containing at least 50,000 samples.",
                "id": "query_0"
            },
            {
                "text": "Need a pre-trained language model suitable for text generation tasks, optimized for low-latency applications.",
                "id": "query_1"
            },
            {
                "text": "Searching for a dataset of medical images annotated with disease labels, including at least 5,000 X-ray images.",
                "id": "query_2"
            },            
        ]

        print("=== Retreving top K documents to each query ===")
        query_loader = Queries(queries=queries).build_loader()
        sem_search_results = self.embedding_store.retrieve_topk_documents(
            self.emb_model, query_loader, topk=topk, save_dirpath=self.topk_dirpath,
            **self.retrieve_topk_documents_func_kwargs
        )

        print("=== Using LLM-as-a-judge principle to evaluate (query, doc) pairs ===")
        evaluate_query_doc_pairs(
            self.llm, query_loader, sem_search_results, 
            text_dirpath=self.asset_text_dirpath, 
            save_dirpath=self.llm_eval_dirpath
        )
        build_query_json_from_llm_eval(
            query_loader.dataset, sem_search_results, 
            self.llm_eval_dirpath, 
            savepath=self.query_jsonfile_path,
            score_function=score_function
        )
        query_loader_with_gt = Queries(json_path=self.query_jsonfile_path).build_loader()

        print("=== Computing retrieval metrics ===")
        all_topk = np.array([3, 5, 10], dtype=np.int32)
        all_topk = all_topk[all_topk <= topk].tolist()
        eval = RetrievalEvaluation(relevance_func=relevance_function)
        eval.evaluate(
            self.emb_model, self.embedding_store, query_loader_with_gt, 
            topk=all_topk,
            load_topk_docs_dirpath=self.topk_dirpath,
            metrics_savepath=self.metrics_savepath,
            retrieve_topk_documents_func_kwargs=self.retrieve_topk_documents_func_kwargs
        )
        print("=========== PRECISION EVALUATION CONCLUDED ===========")