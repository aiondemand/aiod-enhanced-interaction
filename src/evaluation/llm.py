from tqdm import tqdm
import json
import os
from typing import Type, Callable
from enum import Enum
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.language_models.llms import BaseLLM
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI
from torch.utils.data import DataLoader

from dataset import AnnotatedDoc, Queries, QueryDatapoint
from embedding_stores import SemanticSearchResult
from lang_chains import Chain, SimpleChain, ChainOutputOpts


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
            llm = ChatOpenAI(model="gpt-4o")
        if pydantic_model is None:
            pydantic_model = (
                RelevanceEvaluationMultipleDocs
                if compare_multiple_documents_to_a_query
                else RelevanceEvaluation
            )

        if compare_multiple_documents_to_a_query:
            prompt_templates = [cls.system_prompt_multiple_docs, cls.prompt_template_multiple_docs]
        else:
            prompt_templates = [cls.system_prompt_one_doc, cls.prompt_template_one_doc]            

        postprocess_lambda = None
        use_openai_bind_tools = isinstance(llm, BaseChatOpenAI)

        if use_openai_bind_tools:
            postprocess_lambda = lambda out: out[0]
        else:
            prompt_templates[1] += "\n\n{format}"

        chain_output_opts = ChainOutputOpts(
            langchain_parser_class=(
                None if use_openai_bind_tools else JsonOutputParser
            ),
            pydantic_model=pydantic_model,
            schema_placeholder_name="format",
            use_openai_bind_tools=use_openai_bind_tools
        )
        chain_wrapper = SimpleChain(
            llm, prompt_templates, 
            chain_output_opts=chain_output_opts, 
            postprocess_lambda=postprocess_lambda
        )
        return chain_wrapper
    
    @staticmethod
    def build_multiple_document_prompt(documents: list[str]) -> str:
        string_placeholder = "\n### Document {doc_it}:\n{doc}"
        string = ""
        for it, doc in enumerate(documents):
            string += string_placeholder.format(doc_it=it+1, doc=doc)
        
        return string


def evaluate_query_doc_pairs(
    llm: Chain, 
    query_loader: DataLoader, 
    topk_documents: list[SemanticSearchResult],
    text_dirpath: str,
    save_dirpath: str,
    num_attempts: int = 3
) -> None:
    queries: list[QueryDatapoint] = query_loader.dataset.queries
    for query, topk_doc_ids in tqdm(zip(queries, topk_documents), total=len(queries)):
        os.makedirs(os.path.join(save_dirpath, query.id), exist_ok=True)
        
        for doc_id in topk_doc_ids.doc_ids:
            savepath = os.path.join(save_dirpath, query.id, f"{doc_id}.json")
            if os.path.exists(savepath):
                continue
            with open(os.path.join(text_dirpath, f"{doc_id}.txt")) as f:
                doc_text = f.read()

            for _ in range(num_attempts):
                try:
                    pred = llm.invoke({
                        "query": query.text,
                        "document": doc_text
                    })
                    with open(savepath, "w") as f:
                        json.dump(pred, f, ensure_ascii=False)
                    break
                except Exception as e: 
                    print(e)
                    continue
            else:
                print(f"We were unable to evaluate query (id={query.id}) doc (id={doc_id}) pair")


def build_query_json_from_llm_eval(
    dataset: Queries, sem_search: list[SemanticSearchResult], 
    llm_eval_dirpath: str, savepath: str,
    score_function: Callable[[dict], float] | None = None,
) -> None:
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


if __name__ == "__main__":
    from utils import init
    init(return_chroma_client=False)

    QUERY = "I need a multilingual textual dataset that has over 10 000 datapoints for the task of summarization. I'd prefer the dataset to primarily contain news articles."
    doc_ids = ['468', '126982', '3310', '232388', '141388', '720', '46498', '14', '307718', '145626']

    chain = LLM_Evaluator.build_chain()
    with open(f"./data/texts/730.txt") as f:
        document = f.read()
    
    output = chain.invoke({
        "query": QUERY,
        "document": document
    })

    print(output)

    # import os
    # os.makedirs("./temp/llm-eval-test2/docs", exist_ok=True)
    # os.makedirs("./temp/llm-eval-test2/pred", exist_ok=True)

    # from tqdm import tqdm
    # for it, doc_id in tqdm(enumerate(doc_ids)):
    #     with open(f"./data/texts/{doc_id}.txt") as f:
    #         document = f.read()
        
    #     with open(f"./llm-eval-test/docs/{it}.txt", "w") as f:
    #         f.write(document)

    #     output = chain.invoke({
    #         "query": QUERY,
    #         "document": document
    #     })

    #     import json
    #     with open(f"./llm-eval-test/pred/{it}.txt", "w") as f:
    #         json.dump(output, f, ensure_ascii=False)
