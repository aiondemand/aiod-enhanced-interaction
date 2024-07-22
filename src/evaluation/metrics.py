from typing import Callable
import json
import os
import numpy as np
from sklearn.metrics import ndcg_score
from torch.utils.data import DataLoader
from pydantic import BaseModel

from dataset import Queries, QueryDatapoint
from embedding_stores import EmbeddingStore, SemanticSearchResult
from model import EmbeddingModel


class RetrievalMetricsAtK(BaseModel):
    k: int    
    prec: list[float] | float = []
    rec: list[float] | float = []
    AP: list[float] | float = []
    ndcg: list[float] | float = []

    def compute_metrics_for_datapoint(
        self, query_dp: QueryDatapoint, query_results: SemanticSearchResult,
        relevance_func: Callable[[float], bool] | None = None
    ) -> None:
        if relevance_func is None:
            relevance_func = lambda score: score >= 3

        relevant_docs = [
            doc.id for doc in query_dp.get_relevant_documents(relevance_func)
        ]        
        retrieved_docs = query_results.doc_ids
        
        # RETRIEVAL
        self.prec.append(precision_at_k(relevant_docs, retrieved_docs, k=self.k))
        self.rec.append(recall_at_k(relevant_docs, retrieved_docs, k=self.k))
        self.AP.append(average_precision_at_k(relevant_docs, retrieved_docs, k=self.k))

        # RANKING (NDCG)
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


class RetrievalMetrics(BaseModel):
    results_at: dict[str, RetrievalMetricsAtK] | None = None

    def __init__(self, topk: list[int]) -> None:
        super().__init__()
        self.results_at = {
            str(k): RetrievalMetricsAtK(k=k)
            for k in topk
        }

    def average_results(self) -> None:
        for k in self.results_at.keys():
            self.results_at[k].average_results()
    

class SpecificAssetQueriesMetricsAtK(BaseModel):
    k: int
    asset_hit_rate: list[float] | float = []
    asset_position: list[int] | float = []

    def compute_metrics_for_datapoint(self, gt_doc_id: str, relevant_docs: str) -> None:
        indices = np.where(relevant_docs[:self.k] == gt_doc_id)[0]
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
    

class SpecificAssetQueriesMetrics(BaseModel):
    results_at: dict[str, SpecificAssetQueriesMetricsAtK] = None

    def __init__(self, topk: list[int]) -> None:
        super().__init__()
        self.results_at = {
            k: SpecificAssetQueriesMetricsAtK(k=k)
            for k in topk
        }

    def average_results(self) -> None:
        for k in self.results_at.keys():
            self.results_at[k].average_results()


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
        model: EmbeddingModel, 
        embedding_store: EmbeddingStore, 
        query_loader: DataLoader,
        load_topk_docs_dirpath: str | None = None, 
        metrics_savepath: str | None = None,
        topk: list[int] = [3, 5, 10],
        retrieve_topk_documents_func_kwargs: dict | None = None,
    ) -> RetrievalMetrics:
        query_ds: Queries = query_loader.dataset
        if retrieve_topk_documents_func_kwargs is None:
            retrieve_topk_documents_func_kwargs = {}

        sem_search_results = embedding_store.retrieve_topk_documents(
            model, query_loader, topk=max(topk), load_dirpath=load_topk_docs_dirpath,
            **retrieve_topk_documents_func_kwargs
        )

        metrics = RetrievalMetrics(topk)
        for query, query_results in zip(query_ds, sem_search_results):
            for k in topk:
                metrics.results_at[str(k)].compute_metrics_for_datapoint(
                    query_dp=query, 
                    query_results=query_results,
                    relevance_func=self.relevance_func
                )
            
        metrics.average_results()
        if metrics_savepath is not None:
            os.makedirs(os.path.dirname(metrics_savepath), exist_ok=True)
            with open(metrics_savepath, "w") as f:
                json.dump(metrics.model_dump(), f, ensure_ascii=False)
        return metrics

        
class SpecificAssetQueriesEvaluation:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        
    def evaluate(
        self,
        model: EmbeddingModel, 
        embedding_store: EmbeddingStore, 
        query_loader: DataLoader,
        load_topk_docs_dirpath: str | None = None, 
        metrics_savepath: str | None = None,
        topk: list[int] = [3, 5, 10]
    ) -> SpecificAssetQueriesMetrics:
        query_ds: Queries = query_loader.dataset
        sem_search_results = embedding_store.retrieve_topk_documents(
            model, query_loader, topk=max(topk), load_dirpath=load_topk_docs_dirpath
        )

        metrics = SpecificAssetQueriesMetrics(topk)
        for query, query_results in zip(query_ds, sem_search_results):
            gt_doc_id = query.annotated_docs[0].id
            for k in topk:
                metrics.results_at[k].compute_metrics_for_datapoint(
                    gt_doc_id, query_results.doc_ids
                )            
    
        metrics.average_results()
        if metrics_savepath is not None:
            os.makedirs(os.path.dirname(metrics_savepath), exist_ok=True)
            with open(metrics_savepath, "w") as f:
                json.dump(metrics, f, ensure_ascii=False)
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
