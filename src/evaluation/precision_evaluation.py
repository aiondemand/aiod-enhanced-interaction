import os
from typing import Callable, Literal
import numpy as np

from dataset import Queries
from embedding_stores import EmbeddingStore
from evaluation.llm_evaluator import LLM_Evaluator
from evaluation.metrics import RetrievalEvaluation
from evaluation.query_generation import GenericQueryGeneration
from model.lm_encoders.models import EmbeddingModel


class PrecisionEvaluationPipeline:
    # TODO retrieval_model should be later or replaced by the retrieval system abstraction
    def __init__(
        self, embedding_model: EmbeddingModel,
        embedding_store: EmbeddingStore,
        model_name: str,
        asset_type: Literal["dataset", "model", "publication"],
        generate_queries_dirpath: str,
        asset_text_dirpath: str, 
        topk_dirpath: str, 
        llm_eval_dirpath: str,
        annotated_query_dirpath: str,
        metrics_dirpath: str,
        post_process_llm_prediction_function: Callable[[dict], dict] | None = None,
        retrieve_topk_documents_func_kwargs: dict | None = None,
        llm_evaluator_build_chain_kwargs: dict | None = None,
        llm_evaluator_num_docs_to_compare_at_the_time: int = 1
    ) -> None:
        self.embedding_model = embedding_model
        self.embedding_store = embedding_store

        query_gen_chain = GenericQueryGeneration.build_chain(
            asset_type=asset_type,
            query_counts=[10, 30, 50]
        )
        self.query_generation = GenericQueryGeneration(chain=query_gen_chain)

        self.llm_evaluator_build_chain_kwargs = llm_evaluator_build_chain_kwargs
        if llm_evaluator_build_chain_kwargs is None:
            self.llm_evaluator_build_chain_kwargs = {}

        judge_chain = LLM_Evaluator.build_chain(**self.llm_evaluator_build_chain_kwargs)
        multiple_docs = self.llm_evaluator_build_chain_kwargs.get(
            "compare_multiple_documents_to_a_query", False
        )
        self.llm_evaluator_num_doct_to_compare_at_the_time = llm_evaluator_num_docs_to_compare_at_the_time
        if multiple_docs is False:
            self.llm_evaluator_num_doct_to_compare_at_the_time = 1
        self.llm_evaluator = LLM_Evaluator(
            judge_chain, 
            num_docs_to_compare_at_the_time=self.llm_evaluator_num_doct_to_compare_at_the_time
        )
        
        self.model_name = model_name
        self.asset_type = asset_type
        self.generate_queries_dirpath = generate_queries_dirpath
        self.asset_text_dirpath = asset_text_dirpath
        self.topk_dirpath = topk_dirpath
        self.llm_eval_dirpath = llm_eval_dirpath
        self.annotated_query_dirpath = annotated_query_dirpath
        self.metrics_dirpath = metrics_dirpath
        
        self.post_process_llm_prediction_function = post_process_llm_prediction_function
        if post_process_llm_prediction_function is None:
            self.post_process_llm_prediction_function = lambda x: x
        self.retrieve_topk_documents_func_kwargs = retrieve_topk_documents_func_kwargs
        if retrieve_topk_documents_func_kwargs is None:
            self.retrieve_topk_documents_func_kwargs = {}

    
    def execute(
        self, 
        topk: int = 10, 
        score_function: Callable[[dict], float] | None = None,
        relevance_function: Callable[[float], bool] | None = None
    ) -> None:
        print("=========== PRECISION EVALUATION STARTED ===========")

        print("===== Generation of generic queries =====")
        self.query_generation.generate(savedir=self.generate_queries_dirpath)
        query_types = self.query_generation.get_query_types()
    
        for query_type in query_types:
            self._inner_pipeline(
                query_type=query_type,
                topk=topk, 
                score_function=score_function, 
                relevance_function=relevance_function
            )

        print("=========== PRECISION EVALUATION CONCLUDED ===========")

    def _inner_pipeline(self, query_type: str, topk: int = 10, 
        score_function: Callable[[dict], float] | None = None,
        relevance_function: Callable[[float], bool] | None = None
    ) -> None:
        print(f"===== Evaluating '{query_type}' queries =====")

        query_filepath = os.path.join(self.generate_queries_dirpath, f"{query_type}.json")
        topk_dirpath = os.path.join(self.topk_dirpath, self.model_name, query_type)
        llm_eval_dirpath = os.path.join(self.llm_eval_dirpath, query_type)
        annotated_query_savepath = os.path.join(
            self.annotated_query_dirpath, self.model_name, query_type, "queries.json"
        )
        metrics_savepath = os.path.join(
            self.metrics_dirpath, self.model_name, query_type, "results.json"
        )

        print("=== Retreving top K documents to each query ===")
        query_loader = Queries(json_path=query_filepath).build_loader()
        sem_search_results = self.embedding_store.retrieve_topk_document_ids(
            self.embedding_model, query_loader, topk=topk, 
            load_dirpath=topk_dirpath, save_dirpath=topk_dirpath,
            **self.retrieve_topk_documents_func_kwargs
        )

        print("=== Using LLM-as-a-judge principle to evaluate (query, doc) pairs ===")

        self.llm_evaluator.evaluate_query_doc_pairs(
            query_loader, sem_search_results, 
            text_dirpath=self.asset_text_dirpath, 
            save_dirpath=llm_eval_dirpath
        )
        LLM_Evaluator.build_query_json_from_llm_eval(
            query_loader.dataset, sem_search_results, 
            llm_eval_dirpath, 
            savepath=annotated_query_savepath,
            score_function=score_function
        )
        query_loader_with_gt = Queries(json_path=annotated_query_savepath).build_loader()

        print("=== Computing retrieval metrics ===")
        all_topk = np.array([3, 5, 10], dtype=np.int32)
        all_topk = all_topk[all_topk <= topk].tolist()
        eval = RetrievalEvaluation(relevance_func=relevance_function)
        eval.evaluate(
            self.embedding_model, self.embedding_store, query_loader_with_gt, 
            topk=all_topk,
            load_topk_docs_dirpath=topk_dirpath,
            metrics_savepath=metrics_savepath,
            retrieve_topk_documents_func_kwargs=self.retrieve_topk_documents_func_kwargs
        )

if __name__ == "__main__":
    pass