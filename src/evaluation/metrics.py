from __future__ import annotations
from typing import Callable, Type
import json
import os
import numpy as np
from abc import abstractmethod
from sklearn.metrics import ndcg_score
from torch.utils.data import DataLoader

from dataset import Queries
from model.retrieval import RetrievalSystem
from data_types import (
    MetricsClassAtK, MetricsClass, SemanticSearchResult, QueryDatapoint
)


class RetrievalMetricsAtK(MetricsClassAtK):    
    prec: list[float] | float = []
    rec: list[float] | float = []
    AP: list[float] | float = []
    ndcg: list[float] | float = []

    def compute_metrics_for_datapoint(
        self, query_dp: QueryDatapoint, query_results: SemanticSearchResult, k: int,
        compute_precision_only: bool = True,
        relevance_func: Callable[[float], bool] | None = None
    ) -> None:
        if relevance_func is None:
            relevance_func = lambda score: score >= 3

        relevant_docs = [
            doc.id for doc in query_dp.get_relevant_documents(relevance_func)
        ]        
        all_retrieved_docs = query_results.doc_ids
        
        # RETRIEVAL
        self.prec.append(precision_at_k(relevant_docs, all_retrieved_docs, k=k))
        if compute_precision_only is False: 
            self.rec.append(recall_at_k(relevant_docs, all_retrieved_docs, k=k))
            self.AP.append(average_precision_at_k(relevant_docs, all_retrieved_docs, k=k))

        # RANKING (NDCG)
        retrieved_docs = all_retrieved_docs[:k]
        annotated_doc_ids = np.array([doc.id for doc in query_dp.annotated_docs])
        if (np.isin(retrieved_docs, annotated_doc_ids) == False).sum() > 0:
            raise ValueError(
                "One or more retrieved documents don't have corresponding annotations"
            )
        indices = [
            np.where(annotated_doc_ids == doc_id)[0][0] for doc_id in retrieved_docs
        ]
        doc_scores = np.array(
            [query_dp.annotated_docs[idx].score for idx in indices]
        )[None]

        self.ndcg.append(ndcg_score(
            doc_scores,
            np.arange(len(retrieved_docs), 0, -1)[None]
        ))

    def average_results(self) -> None:
        metric_attr_names = ["prec", "rec", "AP", "ndcg"]
        for attr_name in metric_attr_names:
            avg = np.array(getattr(self, attr_name)).mean()
            setattr(self, attr_name, avg)


class RetrievalMetrics(MetricsClass):
    results_in_top: dict[str, RetrievalMetricsAtK] | None = None

    def __init__(self, topk: list[int]) -> None:
        super().__init__()
        self.results_in_top = {
            str(k): RetrievalMetricsAtK()
            for k in topk
        }

    @staticmethod
    def load(data: dict) -> RetrievalMetrics:
        topk = [int(k) for k in data["results_in_top"].keys()]
        metrics_wrapper = RetrievalMetrics(topk)

        for k in topk:
            metrics_wrapper.results_in_top[str(k)] = RetrievalMetricsAtK(
                **data["results_in_top"][str(k)]
            )
        return metrics_wrapper

    def compute_metrics_for_datapoint(
        self, query_dp: QueryDatapoint, query_results: SemanticSearchResult, 
        compute_precision_only: bool = True,
        relevance_func: Callable[[float], bool] | None = None
    ):
        for k in self.results_in_top.keys():
            self.results_in_top[k].compute_metrics_for_datapoint(
                query_dp, query_results, int(k),
                compute_precision_only=compute_precision_only,
                relevance_func=relevance_func
            )

    def average_results(self) -> None:
        for k in self.results_in_top.keys():
            self.results_in_top[k].average_results()
    

class SpecificAssetQueriesMetricsAtK(MetricsClassAtK):
    asset_hit_rate: list[float] | float = []
    asset_position: list[int] | float = []

    def compute_metrics_for_datapoint(
        self, query_dp: QueryDatapoint, query_results: SemanticSearchResult, k: int
    ) -> None:
        gt_doc_id = query_dp.annotated_docs[0].id
        relevant_docs = np.array(query_results.doc_ids)
        
        indices = np.where(relevant_docs[:k] == gt_doc_id)[0]
        if len(indices) == 0:
            self.asset_hit_rate.append(0)
        else:
            self.asset_hit_rate.append(1)
            self.asset_position.append(indices[0])

    def average_results(self) -> None:
        self.asset_hit_rate = (
            (np.array(self.asset_hit_rate) == 1).sum() / len(self.asset_hit_rate)
        )
        self.asset_position = np.array(self.asset_position).mean()
    

class SpecificAssetQueriesMetrics(MetricsClass):
    results_in_top: dict[str, SpecificAssetQueriesMetricsAtK] = None

    def __init__(self, topk: list[int]) -> None:
        super().__init__()
        self.results_in_top = {
            str(k): SpecificAssetQueriesMetricsAtK()
            for k in topk
        }

    @staticmethod
    def load(data: dict) -> SpecificAssetQueriesMetrics:
        topk = [int(k) for k in data["results_in_top"].keys()]
        metrics_wrapper = SpecificAssetQueriesMetrics(topk)

        for k in topk:
            metrics_wrapper.results_in_top[str(k)] = SpecificAssetQueriesMetricsAtK(
                **data["results_in_top"][str(k)]
            )
        return metrics_wrapper

    def compute_metrics_for_datapoint(
        self, query_dp: QueryDatapoint, query_results: SemanticSearchResult
    ) -> None:
        for k in self.results_in_top.keys():
            self.results_in_top[k].compute_metrics_for_datapoint(
                query_dp, query_results, int(k)
            )

    def average_results(self) -> None:
        for k in self.results_in_top.keys():
            self.results_in_top[k].average_results()


class RetrievalEvaluation:
    def __init__(
        self, 
        relevance_func: Callable[[float], bool] | None = None,
        verbose: bool = False
    ) -> None:
        self.relevance_func = relevance_func
        self.verbose = verbose
        
    def evaluate(
        self, 
        retrieval_system: RetrievalSystem,
        query_loader: DataLoader,
        load_topk_docs_dirpath: str | None = None,
        metrics_savepath: str | None = None,
        topk_levels: list[int] = [3, 5, 10],
    ) -> RetrievalMetrics:
        return _generic_evaluation_loop(
            retrieval_system,
            query_loader,
            evaluation_class_type=RetrievalMetrics,
            load_topk_docs_dirpath=load_topk_docs_dirpath,
            metrics_savepath=metrics_savepath,
            topk_levels=topk_levels,
            compute_metrics_for_datapoint_func_kwargs={
                "relevance_func": self.relevance_func,
                "compute_precision_only": True
            }
        )


class SpecificAssetQueriesEvaluation:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        
    def evaluate(
        self,
        retrieval_system: RetrievalSystem,
        query_loader: DataLoader,
        load_topk_docs_dirpath: str | None = None,
        metrics_savepath: str | None = None,
        topk_levels: list[int] = [5, 10, 20, 50, 100],
    ) -> SpecificAssetQueriesMetrics:
        return _generic_evaluation_loop(
            retrieval_system,
            query_loader,
            evaluation_class_type=SpecificAssetQueriesMetrics,
            load_topk_docs_dirpath=load_topk_docs_dirpath,
            metrics_savepath=metrics_savepath,
            topk_levels=topk_levels
        )

    
def _generic_evaluation_loop(
    retrieval_system: RetrievalSystem,
    query_loader: DataLoader,
    evaluation_class_type: Type[MetricsClass],
    load_topk_docs_dirpath: str | None = None,
    metrics_savepath: str | None = None,
    topk_levels: list[int] = [5, 10, 20, 50, 100],
    compute_metrics_for_datapoint_func_kwargs: dict | None = None
) -> MetricsClass:
    query_ds: Queries = query_loader.dataset
    if compute_metrics_for_datapoint_func_kwargs is None:
        compute_metrics_for_datapoint_func_kwargs = {}
    
    sem_search_results = retrieval_system(
        query_loader, 
        retrieve_topk_document_ids_func_kwargs={
            "load_dirpath": load_topk_docs_dirpath
        }
    )
    
    metrics = evaluation_class_type(topk_levels)
    for query, query_results in zip(query_ds, sem_search_results):
        metrics.compute_metrics_for_datapoint(
            query, query_results, **compute_metrics_for_datapoint_func_kwargs
        )

    metrics.average_results()
    if metrics_savepath is not None:
        os.makedirs(os.path.dirname(metrics_savepath), exist_ok=True)
        with open(metrics_savepath, "w") as f:
            json.dump(metrics.model_dump(), f, ensure_ascii=False)
    return metrics


def precision_at_k(relevant: list[str], retrieved: list[str], k: int = 10) -> float:
    relevant = set(relevant)
    retrieved_at_k = set(retrieved[:k])
    precision = len(relevant.intersection(retrieved_at_k)) / k if k > 0 else 0
    return precision


def recall_at_k(relevant: list[str], retrieved: list[str], k: int = 10) -> float:
    relevant = set(relevant)
    retrieved_at_k = set(retrieved[:k])

    return (
        len(relevant.intersection(retrieved_at_k)) / len(relevant)
        if len(relevant) > 0 else 0
    )


def average_precision_at_k(
    relevant: list[str], retrieved: list[str], k: int = 10
) -> float:
    relevant = set(relevant)
    precision_sum = 0
    num_relevant_docs = len(relevant)

    for i in range(k):
        if retrieved[i] in relevant:
            precision_sum += precision_at_k(relevant, retrieved, i+1)

    avg_precision = (
        precision_sum / num_relevant_docs 
        if num_relevant_docs > 0 else 0
    )
    return avg_precision


if __name__ == "__main__":
    p = "data/results/hit_rate/gte_large--basic/least_descriptive-long_description_few_tags/results.json"
    with open(p) as f:
        data = json.load(f)

    metrics = SpecificAssetQueriesMetrics.load(data)
    metrics