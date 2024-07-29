# TODO Rag implementation
# start off with general implementation, RAG with ChromaDB
# Once its done, we may want to further extract RAG specific code to a seperate file
import torch
from torch.utils.data import DataLoader
from langchain_core.language_models.llms import BaseLLM

from embedding_stores import EmbeddingStore
from model.base import RetrievalSystem, EmbeddingModel
from data_types import SemanticSearchResult

class RAG_Pipeline(torch.nn.Module, RetrievalSystem):
    # TODO create prompt for the LLM
    # start off with simple retrieval of the best documents matching to the query

    def __init__(
        self, embedding_model: EmbeddingModel, embedding_store: EmbeddingStore, 
        llm: BaseLLM, emb_collection_name: str, document_collection_name: str, 
        topk: int = 100
    ) -> None:
        super().__init__()

        self.embedding_model = embedding_model
        self.embedding_store = embedding_store
        self.llm = llm
        self.emb_collection_name = emb_collection_name
        self.document_collection_name = document_collection_name
        self.topk = topk

    def forward(
        self, query_loader: DataLoader, 
        retrieve_topk_document_ids_func_kwargs: dict | None = None,
        translate_documents_func_kwargs: dict | None = None
    ) -> list[SemanticSearchResult]:
        if retrieve_topk_document_ids_func_kwargs is None:
            retrieve_topk_document_ids_func_kwargs = {}
        if translate_documents_func_kwargs is None: 
            translate_documents_func_kwargs = {}

        semantic_search_results = self.embedding_store.retrieve_topk_document_ids(
            self.embedding_model, query_loader, topk=self.topk, 
            emb_collection_name=self.emb_collection_name,
            **retrieve_topk_document_ids_func_kwargs
        )
        topk_documents = self.embedding_store.translate_sem_results_to_documents(
            semantic_search_results, 
            document_collection_name=self.document_collection_name
            **translate_documents_func_kwargs
        )
        
        # TODO put the documents into the prompt as a knowledge base
        topk_documents
        

class EmbeddingModel_Pipeline(torch.nn.Module, RetrievalSystem):
    def __init__(
        self, embedding_model: EmbeddingModel, embedding_store: EmbeddingStore,
        emb_collection_name: str, topk: int = 10
    ) -> None:
        super().__init__()
        
        self.embedding_model = embedding_model
        self.embedding_store = embedding_store
        self.emb_collection_name = emb_collection_name
        self.topk = topk

    def forward(
        self, query_loader: DataLoader, 
        retrieve_topk_document_ids_func_kwargs: dict | None = None
    ) -> list[SemanticSearchResult]:
        if retrieve_topk_document_ids_func_kwargs is None:
            retrieve_topk_document_ids_func_kwargs = {}

        return self.embedding_store.retrieve_topk_document_ids(
            self.embedding_model, query_loader, topk=self.topk, 
            emb_collection_name=self.emb_collection_name,
            **retrieve_topk_document_ids_func_kwargs
        )
