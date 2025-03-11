from tqdm import tqdm
import json
import os
from typing import Type, Callable
from enum import Enum
from pydantic import BaseModel, Field
from langchain_community.callbacks import get_openai_callback
from langchain_core.language_models.llms import BaseLLM
from torch.utils.data import DataLoader

from dataset import Queries
from lang_chains import SimpleChain, LLM_Chain, load_llm
from data_types import AnnotatedDoc, QueryDatapoint, SemanticSearchResult


class MetStatus(str, Enum):
    TRUE = "true"
    FALSE = "false"
    CANT_TELL = "cannot tell"


class ConditionEvaluation(BaseModel):
    condition: str = Field(..., description="The specific user-defined condition/constraint being evaluated.")
    details: str = Field(..., description="A brief description of how the condition was or was not met.")
    mandatory: bool = Field(..., description="Whether this condition is mandatory. It is usually mandatory unless specified otherwise")
    met: MetStatus = Field(..., description="Whether the condition was met (true/false/cannot tell).")


class RelevanceExplanation(BaseModel):
    condition_evaluations: list[ConditionEvaluation] = Field(..., description="A list of condition evaluations.")
    overall_match: str = Field(..., description="A summary statement of the overall relevance of the document to the query based on the met / not met conditions.")
    relevance_rating: int = Field(..., description="A relevance rating on a scale from 1 to 5 based on previous performed condition evaluations")


class RelevanceEvaluation(BaseModel):
    explanation: RelevanceExplanation = Field(..., description="Detailed explanation of the document relevance to the query.")


class RelevanceEvaluationOneDocExplicitQueryDoc(BaseModel):
    query: str = Field(..., description="The user query describing the requirements and desired properties of an asset.")
    document: str = Field(..., description="The document (a short asset description together with additional metadata) being evaluated.")
    explanation: RelevanceExplanation = Field(..., description="Detailed explanation of the document relevance to the query.")


class RelevanceEvaluationMultipleDocsExplicitQueryDoc(BaseModel):
    relevances: list[RelevanceEvaluationOneDocExplicitQueryDoc] = Field(..., description="Evaluation of relevance of each document/asset to an user query.")


class RelevanceEvaluationMultipleDocs(BaseModel):
    relevances: list[RelevanceEvaluation] = Field(..., description="Evaluation of relevance of each document/asset to an user query.")


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
                - Met: Whether the condition was met (true/false/cannot tell).

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
                - Met: Whether the condition was met (true/false/cannot tell).

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
        compare_multiple_documents_to_a_query: bool = False,
        generating_explicit_query_doc_pairs: bool = False
    ) -> SimpleChain:
        if llm is None:
            llm = load_llm()
        if pydantic_model is None:
            if compare_multiple_documents_to_a_query and generating_explicit_query_doc_pairs:
                pydantic_model = RelevanceEvaluationMultipleDocsExplicitQueryDoc
            elif compare_multiple_documents_to_a_query:
                pydantic_model = RelevanceEvaluationMultipleDocs
            else:
                pydantic_model = RelevanceEvaluation

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

    def __init__(
        self, chain: SimpleChain | None = None,
        num_docs_to_compare_at_the_time: int = 1
    ) -> None:
        if chain is None:
            self.chain = self.build_chain()
            self.num_docs_to_compare_at_the_time = 1
        else:
            self.chain = chain
            self.num_docs_to_compare_at_the_time = num_docs_to_compare_at_the_time

    def __call__(self, input: dict) -> dict | str | None:
        return self.chain.invoke(input)

    def evaluate_query_doc_pairs(
        self,
        query_loader: DataLoader,
        topk_documents: list[SemanticSearchResult],
        text_dirpath: str,
        save_dirpath: str
    ) -> None:
        queries: list[QueryDatapoint] = query_loader.dataset.queries

        print(f"...Evaluating relevance of retrieved documents to {len(queries)} queries...")
        with get_openai_callback() as cb:
            for query, topk_doc_ids in tqdm(zip(queries, topk_documents), total=len(queries)):
                os.makedirs(os.path.join(save_dirpath, query.id), exist_ok=True)
                remaining_doc_ids = [
                    doc_id
                    for doc_id in topk_doc_ids.doc_ids
                    if os.path.exists(
                        os.path.join(save_dirpath, query.id, f"{doc_id}.json")
                    ) == False
                ]
                if len(remaining_doc_ids) == 0:
                    continue

                model_predictions = []
                doc_groups = self._group_docs(text_dirpath, remaining_doc_ids)
                group_size = self.num_docs_to_compare_at_the_time
                for it, doc_group in enumerate(doc_groups):
                    doc_ids = remaining_doc_ids[
                        it*group_size: (it+1)*group_size
                    ]
                    model_predictions = self.calc_doc_relevance(query, doc_group)
                    self.save_doc_relevance(query, doc_ids, model_predictions, save_dirpath)

            # print(cb)

    def _group_docs(self, text_dirpath: str, doc_ids: list[str]) -> list[list[str]]:
        groups = []
        for i in range(0, len(doc_ids), self.num_docs_to_compare_at_the_time):
            group = []
            ids = doc_ids[i: i+self.num_docs_to_compare_at_the_time]
            for doc_id in ids:
                with open(os.path.join(text_dirpath, f"{doc_id}.txt")) as f:
                    group.append(f.read())
            groups.append(group)

        return groups

    def calc_doc_relevance(
        self, query: QueryDatapoint, docs: list[str]
    ) -> list[dict | None]:
        if self.num_docs_to_compare_at_the_time == 1:
            return [self({
                "query": query.text,
                "document": docs[0]
            })]

        multiple_docs_str = self.build_multiple_document_prompt(docs)
        multiple_pred = self({
            "query": query.text,
            "multiple_documents": multiple_docs_str
        })
        if len(multiple_pred["relevances"]) != len(docs):
            return [None for _ in range(len(docs))]
        return [
            { "explanation": p["explanation"] }
            for p in multiple_pred["relevances"]
        ]

    def save_doc_relevance(
        self, query: QueryDatapoint, doc_ids: list[str],
        predictions: list[dict | None], savedir: str
    ) -> None:
        for doc_id, pred in zip(doc_ids, predictions):
            if pred is not None:
                savepath = os.path.join(savedir, query.id, f"{doc_id}.json")
                with open(savepath, "w") as f:
                    json.dump(pred, f, ensure_ascii=False)
            else:
                print(f"We were unable to evaluate query (id={query.id}) doc (id={doc_id}) pair")

    @classmethod
    def build_multiple_document_prompt(cls, documents: list[str]) -> str:
        string_placeholder = "### Asset {doc_it} to evaluate to the user query:\n{doc}\n\n"
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
        if os.path.exists(savepath):
            return
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


    @staticmethod
    def heuristic_score_function(llm_output: dict) -> float:
        filters = RelevanceEvaluation(**llm_output).explanation.condition_evaluations
        req_filters = [cond for cond in filters if cond.mandatory]
        opt_filters = [cond for cond in filters if cond.mandatory is False]

        required_score = sum([
            f.met == MetStatus.TRUE for f in req_filters
        ]) / len(req_filters)

        optional_score = 0
        if len(opt_filters) > 0:
            optional_score = sum([
                f.met == MetStatus.TRUE for f in opt_filters
            ]) / len(opt_filters)

        return required_score + optional_score*(required_score)**2
