# # TODO Rag implementation
# # start off with general implementation, RAG with ChromaDB
# # Once its done, we may want to further extract RAG specific code to a seperate file
# import torch
# from langchain_core.language_models.llms import BaseLLM

# from dataset import Queries, QueryDatapoint
# from embedding_stores import EmbeddingStore
# from model.base import RetrievalSystem, EmbeddingModel

# class RAG_Pipeline(torch.nn.Module, RetrievalSystem):
#     # TODO create prompt for the LLM
#     # start off with simple retrieval of the best documents matching to the query

#     def __init__(
#         self, embedding_model: EmbeddingModel, embedding_store: EmbeddingStore, 
#         llm: BaseLLM, topk: int = 100, 
#         additional_translate_documents_kwargs: dict | None = None
#     ) -> None:
#         super().__init__()

#         self.embedding_model = embedding_model
#         self.embedding_store = embedding_store
#         self.llm = llm
#         self.topk = topk

#         self.additional_translate_documents_kwargs = additional_translate_documents_kwargs
#         if additional_translate_documents_kwargs: 
#             self.additional_translate_documents_kwargs = {}

#     def forward(self, queries: list[QueryDatapoint]) -> list[list[str]]:
#         query_loader = Queries(queries=queries).build_loader()
#         semantic_search_results = self.embedding_store.retrieve_topk_document_ids(
#             self.embedding_model, query_loader, topk=self.topk
#         )
#         topk_documents = self.embedding_store.translate_results_to_documents(
#             semantic_search_results, **self.additional_translate_documents_kwargs
#         )
        
#         # TODO put the documents into the prompt as a knowledge base
#         topk_documents
        

# class EmbeddingModel_Pipeline(RetrievalSystem):
#     # TODO integrate into prior pipelines => namely into precision, recall pipeline evaluations
    
#     def __init__(
#         self, embedding_model: EmbeddingModel, embedding_store: EmbeddingStore,
#         topk: int = 10, retrieve_topk_document_ids_func_kwargs: dict | None = None
#     ) -> None:
#         super().__init__()

#         self.embedding_model = embedding_model
#         self.embedding_store = embedding_store
#         self.topk = topk

#         self.retrieve_topk_document_ids_func_kwargs = retrieve_topk_document_ids_func_kwargs
#         if retrieve_topk_document_ids_func_kwargs is None:
#             self.retrieve_topk_document_ids_func_kwargs = {}

#     def forward(self, queries: list[QueryDatapoint]) -> list[list[str]]:
#         query_loader = Queries(queries=queries).build_loader()
#         semantic_search_results = self.embedding_store.retrieve_topk_document_ids(
#             self.embedding_model, query_loader, topk=self.topk
#         )
#         return [
#             results.doc_ids for results in semantic_search_results
#         ]
