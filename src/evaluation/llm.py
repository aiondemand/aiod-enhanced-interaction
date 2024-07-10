from typing import Type
from enum import Enum
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.language_models.llms import BaseLLM
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI

from lang_chains import SimpleChain, ChainOutputOpts


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
    query: str = Field(..., description="The user query describing the requirements and desired properties of an asset.")
    document: str = Field(..., description="The document (a short asset description together with additional metadata) being evaluated.")
    explanation: RelevanceExplanation = Field(..., description="Detailed explanation of the document relevance to the query.")
    

class LLM_Evaluation:
    system_prompt = """
        You are an expert evaluator tasked with assessing the relevance of machine learning assets (such as models or datasets) to specific user queries. Each query describes the requirements and desired properties of an asset. You will be given a query and a corresponding document (a short asset description together with additional metadata) and are asked to provide the following:

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
    prompt_template = """
        Query: {query}
        Document: {document}
    """

    @classmethod
    def build_chain(
        cls, llm: BaseLLM | None = None, pydantic_model: Type[BaseModel] | None = None
    ) -> SimpleChain:
        if llm is None:
            llm = ChatOpenAI(model="gpt-4o")
        if pydantic_model is None:
            pydantic_model = RelevanceEvaluation

        prompt_templates = [cls.system_prompt, cls.prompt_template]
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


if __name__ == "__main__":
    from utils import init
    init(return_chroma_client=False)

    QUERY = "I need a multilingual textual dataset that has over 10 000 datapoints for the task of summarization. I'd prefer the dataset to primarily contain news articles."
    doc_ids = ['468', '126982', '3310', '232388', '141388', '720', '46498', '14', '307718', '145626']

    # llm = Ollama(model="mistral", num_predict=1024)
    # chain = LLM_Evaluation._build_chain(
    #     llm, pydantic_model=RelevanceEvaluation
    # )

    chain = LLM_Evaluation.build_chain()

    with open(f"./data/texts/720.txt") as f:
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
