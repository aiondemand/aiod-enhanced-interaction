from typing import Callable, Literal, Type
import torch
import numpy as np
from torch.utils.data import DataLoader
from langchain_core.language_models.llms import BaseLLM
from langchain.pydantic_v1 import BaseModel, Field
from langchain_community.callbacks import get_openai_callback
from time import time

from tqdm import tqdm

from embedding_stores import EmbeddingStore, LocalTopKDocumentsStore
from lang_chains import SimpleChain, LLM_Chain, load_llm
from model.base import RetrievalSystem, EmbeddingModel
from data_types import QueryDatapoint, RetrievedDocuments, SemanticSearchResult


class TopAssetIds(BaseModel):
    document_ids: list[str] = Field(..., description="List of IDs of the most relevant documents to the user query.")
    

class RAG_Pipeline(torch.nn.Module, RetrievalSystem):
    llm_system_prompt_basic = """
        You are an advanced language model tasked with evaluating the relevance of 
        a set of machine learning asset descriptions to a specific user query. 
        Given the query and a list of {doc_type}s, your goal is to output the list of 
        IDs of the most relevant {doc_type}s to that query.
    """
    
    llm_system_prompt = """
        You are an advanced language model tasked with evaluating the relevance of 
        a set of machine learning asset descriptions to a specific user query. 
        Given the query and a list of {doc_type}s, your goal is to:

        - Evaluate Relevance: Assess whether each {doc_type} is relevant to the query.
        - Check Conditions: Check whether all conditions mentioned in the query are met in the {doc_type} description.
        - Select Top {output_count}: Select the top {output_count} {doc_type} that best match the query in terms of relevance and condition fulfillment.
        - Rank the Documents: Rank the selected {doc_type} from most to least relevant.

    """
    llm_system_prompt_steps = """   
        ### Steps to take:
        1) Evaluate each asset based on the factors listed above.
        2) Assign a relevance score to each asset on a scale from 1 to 10, with 10 being the most relevant.
        3) Select the top {output_count} assets with the highest relevance scores.
        4) Ensure the selected assets collectively cover all aspects of the query comprehensively.
        5) Return only the IDs of the top {output_count} assets.

    """
    llm_system_prompt_output = """
        Proceed with the evaluation, selection, and ranking based on the following provided query and a list of {doc_type}s, 
        and return the IDs of the top {output_count} most relevant {doc_type}s in the specified format.
    """
    
    user_prompt = """
        # Query: 
        {query}

        # {doc_type}s to evaluate:
        {multiple_docs}

        IDs of top {output_count} most relevant {doc_type}s:
    """

    def __init__(
        self, embedding_model: EmbeddingModel, embedding_store: EmbeddingStore, 
        emb_collection_name: str, document_collection_name: str, 
        stringify_document_func: Callable[[dict], str], 
        retrieval_topk: int = 100, output_topk: int = 10,
        llm: BaseLLM | None = None, pydantic_model: Type[BaseModel] | None = None,
        prompt_variation: Literal["basic", "normal", "with_steps"] = "normal", 
        doc_type_name: Literal["dataset", "model"] = "dataset", 
        
    ) -> None:
        super().__init__()

        self.embedding_model = embedding_model
        self.embedding_store = embedding_store
        self.llm_chain = self.build_llm_for_document_validation(
            llm=llm, 
            pydantic_model=pydantic_model, 
            prompt_variation=prompt_variation, 
            doc_type_name=doc_type_name,
            num_output_docs=output_topk
        )
        
        self.emb_collection_name = emb_collection_name
        self.document_collection_name = document_collection_name
        self.stringify_document_func = stringify_document_func
        self.retrieval_topk = retrieval_topk
        self.topk = output_topk
        self.doc_type_name = doc_type_name

    def forward(
        self, query_loader: DataLoader, 
        retrieve_topk_document_ids_func_kwargs: dict | None = None,
        translate_documents_func_kwargs: dict | None = None
    ) -> list[SemanticSearchResult]:
        if retrieve_topk_document_ids_func_kwargs is None:
            retrieve_topk_document_ids_func_kwargs = {}
        if translate_documents_func_kwargs is None:
            translate_documents_func_kwargs = {}
        
        load_dirpaths = retrieve_topk_document_ids_func_kwargs.pop("load_dirpaths", None)
        save_dirpath = retrieve_topk_document_ids_func_kwargs.pop("save_dirpath", None)

        if load_dirpaths is not None:
            try:
                topk_store = LocalTopKDocumentsStore(
                    load_dirpaths, topk=self.topk
                )
                return topk_store.load_topk_documents(query_loader)
            except:
                pass

        print("... [RAG] Retrieving docs ...")
        semantic_results, retrieved_doc_objs = self._retrieve_documents(
            query_loader, 
            retrieve_topk_document_ids_func_kwargs, 
            translate_documents_func_kwargs
        )
        print("... [RAG] Generating LLM output ...")
        top_documents = self._generate_output(
            query_loader, 
            semantic_results, 
            retrieved_doc_objs
        )

        if save_dirpath is not None:
            topk_store = LocalTopKDocumentsStore(save_dirpath, topk=self.topk)
            topk_store.store_topk_documents(top_documents)
        return top_documents
    
    def _retrieve_documents(
        self, query_loader: DataLoader, 
         retrieve_topk_document_ids_func_kwargs: dict | None = None,
        translate_documents_func_kwargs: dict | None = None
    ) -> tuple[list[SemanticSearchResult], list[RetrievedDocuments]]:
        if retrieve_topk_document_ids_func_kwargs is None:
            retrieve_topk_document_ids_func_kwargs = {}
        if translate_documents_func_kwargs is None: 
            translate_documents_func_kwargs = {}

        semantic_search_results = self.embedding_store.retrieve_topk_document_ids(
            self.embedding_model, query_loader, topk=self.retrieval_topk, 
            emb_collection_name=self.emb_collection_name,
            **retrieve_topk_document_ids_func_kwargs
        )
        retrieved_doc_objs = self.embedding_store.translate_sem_results_to_documents(
            semantic_search_results, 
            document_collection_name=self.document_collection_name,
            **translate_documents_func_kwargs
        )
        return semantic_search_results, retrieved_doc_objs
    
    def _generate_output(
        self, 
        query_loader: DataLoader, 
        semantic_search_results: list[SemanticSearchResult],
        retrieved_documents: list[RetrievedDocuments],
        retry_count_wrong_doc_ids: int = 3
    ) -> list[SemanticSearchResult]:
        queries: list[QueryDatapoint] = query_loader.dataset.queries
        
        final_retrieved_doc_ids = []

        with get_openai_callback() as cb:
            for query, sem_results, context_docs in tqdm(
                zip(queries, semantic_search_results, retrieved_documents),
                total=len(queries)
            ):
                multiple_doc_string = self._build_multiple_docs_prompt(
                    context_docs.document_objects
                )

                for _ in range(retry_count_wrong_doc_ids):
                    try:
                        out = self.llm_chain.invoke({
                            "query": query.text,
                            "multiple_docs": multiple_doc_string,
                            "doc_type": self.doc_type_name,
                            "output_count": self.topk
                        }) 
                        valid_pred_ids = np.isin(
                            np.array(out["document_ids"]), np.array(sem_results.doc_ids)
                        )
                        if (valid_pred_ids == False).sum() > 0 or len(out) != self.topk:
                            raise ValueError("Invalid LLM predictions")

                        final_retrieved_doc_ids.append(SemanticSearchResult(
                            query_id=query.id,
                            doc_ids=out["document_ids"]
                        ))        
                        break
                    except:
                        continue
                else:
                    raise ValueError(f"Invalid LLM predictions for query id=f'{query.id}'")
            print(cb)
            
        return final_retrieved_doc_ids

    def _build_multiple_docs_prompt(self, retrieved_docs: list[dict]) -> str:
        string_placeholder = "### {doc_type} (ID='{doc_id}') to evaluate to the user query:\n{doc}\n\n"
        string = ""
        for doc in retrieved_docs:
            doc_string = self.stringify_document_func(doc)
            string += string_placeholder.format(
                doc_type=self.doc_type_name, 
                doc_id=doc["identifier"],
                doc=doc_string
            )

        return string
    
    @classmethod
    def build_llm_for_document_validation(
        cls, llm: BaseLLM | None = None, 
        pydantic_model: Type[BaseModel] | None = None,
        prompt_variation: Literal["basic", "normal", "with_steps"] = "normal", 
        doc_type_name: Literal["dataset", "model"] = "dataset", 
        num_output_docs: int = 10
    ) -> SimpleChain:
        if llm is None:
            llm = load_llm()
        if pydantic_model is None:
            pydantic_model = TopAssetIds

        if prompt_variation == "basic":
            system_prompt = cls.llm_system_prompt_basic
        elif prompt_variation == "normal":
            system_prompt = cls.llm_system_prompt + cls.llm_system_prompt_output
        elif "with_steps":
            system_prompt = (
                cls.llm_system_prompt + 
                cls.llm_system_prompt_steps + 
                cls.llm_system_prompt_output
            )

        prompt_templates = [
            system_prompt.format(
                doc_type=doc_type_name,
                output_count=num_output_docs
            ), 
            cls.user_prompt
        ]
        return LLM_Chain.build_simple_chain(
            pydantic_model=pydantic_model, 
            prompt_templates=prompt_templates, 
            llm=llm
        )
        

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
