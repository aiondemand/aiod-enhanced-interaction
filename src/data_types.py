from abc import abstractmethod
from typing import Callable, TypeAlias
from pydantic import BaseModel
from pymilvus import MilvusClient
from chromadb.api.client import Client as ChromaClient


VectorDbClient: TypeAlias = ChromaClient | MilvusClient


class AnnotatedDoc(BaseModel):
    id: str
    score: int | float


class QueryDatapoint(BaseModel):
    text: str
    id: str | None = None
    annotated_docs: list[AnnotatedDoc] | None = None

    def get_relevant_documents(
        self, relevance_func: Callable[[float], bool]
    ) -> list[AnnotatedDoc]:
        return [
            doc for doc in self.annotated_docs
            if relevance_func(doc.score)
        ]
    

class SemanticSearchResult(BaseModel):
    query_id: str
    doc_ids: list[str]
    distances: list[float] | None = None


class RetrievedDocuments(BaseModel):
    query_id: str
    document_objects: list[dict]


class MetricsClassAtK(BaseModel):
    @abstractmethod
    def compute_metrics_for_datapoint(
        self, query_dp: QueryDatapoint, query_results: SemanticSearchResult,  
        k: int, **kwargs
    ) -> None:
        pass
        
    @abstractmethod
    def average_results(self) -> None:
        pass


class MetricsClass(BaseModel):
    results_in_top: dict[str, MetricsClassAtK] | None = None

    @abstractmethod
    def compute_metrics_for_datapoint(
        self, query_dp: QueryDatapoint, query_results: SemanticSearchResult, **kwargs
    ) -> None:
        pass

    @abstractmethod
    def average_results(self) -> None:
        pass
    