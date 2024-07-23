from tqdm import tqdm
import json
import os
from typing import Type, Callable
from enum import Enum
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.language_models.llms import BaseLLM
from torch.utils.data import DataLoader

from dataset import AnnotatedDoc, Queries, QueryDatapoint
from embedding_stores import SemanticSearchResult
from lang_chains import Chain, SimpleChain
from evaluation.llm import LLM_Chain, get_default_llm


class MetStatus(str, Enum):
    TRUE = "true"
    FALSE = "false"
    CANT_TELL = "cant tell"
    

class ConditionEvaluation(BaseModel):
    condition: str = Field(..., description="The specific user-defined condition/constraint being evaluated.")
    details: str = Field(..., description="A brief description of how the condition was or was not met.")
    mandatory: bool = Field(True, description="Whether this condition is mandatory. It is usually mandatory unless specified otherwise")
    met: MetStatus = Field(..., description="Whether the condition was met (true/false/cant tell).")


class RelevanceExplanation(BaseModel):
    condition_evaluations: list[ConditionEvaluation] = Field(..., description="A list of condition evaluations.")
    overall_match: str = Field(..., description="A summary statement of the overall relevance of the document to the query based on the met / not met conditions.")
    relevance_rating: int = Field(..., description="A relevance rating on a scale from 1 to 5 based on previous performed condition evaluations")


class RelevanceEvaluation(BaseModel):
    explanation: RelevanceExplanation = Field(..., description="Detailed explanation of the document relevance to the query.")


class RelevanceEvaluationOneDoc(BaseModel):
    query: str = Field(..., description="The user query describing the requirements and desired properties of an asset.")
    document: str = Field(..., description="The document (a short asset description together with additional metadata) being evaluated.")
    explanation: RelevanceExplanation = Field(..., description="Detailed explanation of the document relevance to the query.")


class RelevanceEvaluationMultipleDocs(BaseModel):
    relevances: list[RelevanceEvaluationOneDoc] = Field(..., description="Evaluation of relevance of each document/asset to an user query.")
    

class LLM_Evaluator:
    system_prompt_one_doc = """
        You are an expert evaluator tasked with assessing the relevance of machine learning assets 
        (such as models or datasets) to specific user queries. Each query describes the requirements 
        and desired properties of an asset. You will be given a query and a corresponding document 
        (a short asset description together with additional metadata) and are asked to provide the following:

        1) A detailed explanation structured into the following sections:

            - Condition Evaluations: For each key condition or constraint mentioned in the query, provide:
                - Condition: The specific condition being evaluated.
                - Details: A brief description of how the condition was or was not met.
                - Mandatory: Whether the condition is mandatory (by default it is, unless specified otherwise)
                - Met: Whether the condition was met (true/false/cant tell).
                
            - Overall Match: A summary statement of the overall relevance of the document to the query.

        2) A relevance rating on a scale from 1 to 5, where:
            - 1 = Not relevant at all
            - 2 = Slightly relevant
            - 3 = Moderately relevant
            - 4 = Very relevant
            - 5 = Extremely relevant
    """
    system_prompt_multiple_docs = """
        You are an expert evaluator tasked with assessing the relevance of machine learning assets 
        (such as models or datasets) to specific user queries. Each query describes the requirements 
        and desired properties of an asset. For each evaluation, you will be provided with a query 
        and a set of documents (each containing a short asset description together with additional metadata).
        You need to provide the following for each document:

        1) A detailed explanation structured into the following sections:

            - Condition Evaluations: For each key condition or constraint mentioned in the query, provide:
                - Condition: The specific condition being evaluated.
                - Details: A brief description of how the condition was or was not met.
                - Mandatory: Whether the condition is mandatory (by default it is, unless specified otherwise)
                - Met: Whether the condition was met (true/false/cant tell).
                
            - Overall Match: A summary statement of the overall relevance of the document to the query.

        2) A relevance rating on a scale from 1 to 5, where:
            - 1 = Not relevant at all
            - 2 = Slightly relevant
            - 3 = Moderately relevant
            - 4 = Very relevant
            - 5 = Extremely relevant
    """
    
    prompt_template_one_doc = """
        ### Query: 
        {query}
        ### Document: 
        {document}
    """
    prompt_template_multiple_docs = """
        ### Query: 
        {query}
        {multiple_documents}
    """

    @classmethod
    def build_chain(
        cls, llm: BaseLLM | None = None, pydantic_model: Type[BaseModel] | None = None,
        compare_multiple_documents_to_a_query: bool = False
    ) -> SimpleChain:
        if llm is None:
            llm = get_default_llm()
        if pydantic_model is None:
            pydantic_model = (
                RelevanceEvaluationMultipleDocs
                if compare_multiple_documents_to_a_query
                else RelevanceEvaluation
            )
        prompt_templates = (
            [cls.system_prompt_multiple_docs, cls.prompt_template_multiple_docs]
            if compare_multiple_documents_to_a_query
            else [cls.system_prompt_one_doc, cls.prompt_template_one_doc]
        )

        return LLM_Chain.build_simple_chain(
            pydantic_model=pydantic_model, 
            prompt_templates=prompt_templates, 
            llm=llm
        )
    
    def __init__(self, chain: SimpleChain | None = None) -> None:
        self.chain = chain
        if chain is None:
            self.chain = self.build_chain()

    def evaluate_query_doc_pairs(
        self,
        query_loader: DataLoader, 
        topk_documents: list[SemanticSearchResult],
        text_dirpath: str,
        save_dirpath: str
    ) -> None:
        queries: list[QueryDatapoint] = query_loader.dataset.queries
        
        print(f"...Evaluating relevance of retrieved documents to {len(queries)} queries...")
        for query, topk_doc_ids in tqdm(zip(queries, topk_documents), total=len(queries)):
            os.makedirs(os.path.join(save_dirpath, query.id), exist_ok=True)
            
            for doc_id in topk_doc_ids.doc_ids:
                savepath = os.path.join(save_dirpath, query.id, f"{doc_id}.json")
                if os.path.exists(savepath):
                    continue
                with open(os.path.join(text_dirpath, f"{doc_id}.txt")) as f:
                    doc_text = f.read()
        
                pred = self.chain.invoke({
                    "query": query.text,
                    "document": doc_text
                })
                if pred is not None:
                    with open(savepath, "w") as f:
                        json.dump(pred, f, ensure_ascii=False)
                else:
                    print(f"We were unable to evaluate query (id={query.id}) doc (id={doc_id}) pair")
            
    @staticmethod
    def build_multiple_document_prompt(documents: list[str]) -> str:
        string_placeholder = "\n### Document {doc_it}:\n{doc}"
        string = ""
        for it, doc in enumerate(documents):
            string += string_placeholder.format(doc_it=it+1, doc=doc)
        
        return string

    @staticmethod
    def build_query_json_from_llm_eval(
        dataset: Queries, sem_search: list[SemanticSearchResult], 
        llm_eval_dirpath: str, savepath: str,
        score_function: Callable[[dict], float] | None = None,
    ) -> None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        json_query_datapoints = []
        
        if score_function is None:
            score_function = lambda obj: obj["explanation"]["relevance_rating"]
        for query_topk_docs in sem_search:
            query = dataset.get_by_id(query_topk_docs.query_id)
            doc_ids = query_topk_docs.doc_ids

            annotated_docs = []
            for doc_id in doc_ids:
                p = os.path.join(llm_eval_dirpath, query.id, f"{doc_id}.json")
                with open(p) as f:
                    data = json.load(f)

                annotated_docs.append(AnnotatedDoc(
                    id=doc_id, score=score_function(data)
                ))
                
            json_query_datapoints.append(
                QueryDatapoint(
                    text=query.text, 
                    id=query.id, 
                    annotated_docs=annotated_docs
                ).model_dump()
            )
        with open(savepath, "w") as f:
            json.dump(json_query_datapoints, f, ensure_ascii=False)
