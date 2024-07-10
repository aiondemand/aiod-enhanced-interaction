from typing import Type
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.language_models.llms import BaseLLM
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.llms import Ollama

from lang_chains import SimpleChain, ChainOutputOpts


class MetStatus(str, Enum):
    TRUE = "true"
    FALSE = "false"
    CANT_TELL = "cant tell"


class ConditionEvaluation(BaseModel):
    condition: str = Field(..., description="The specific user-defined condition/constraint being evaluated.")
    met: MetStatus = Field(..., description="Whether the condition was met (true/false/unknown).")
    details: str = Field(..., description="A brief description of how the condition was or was not met.")


class RelevanceExplanation(BaseModel):
    condition_evaluations: list[ConditionEvaluation] = Field(..., description="A list of condition evaluations.")
    overall_match: str = Field(..., description="A summary statement of the overall relevance of the document to the query based on the met / not met conditions.")


class RelevanceEvaluation(BaseModel):
    query: str = Field(..., description="The user query describing the requirements and desired properties of an asset.")
    document: str = Field(..., description="The document (a short asset description together with additional metadata) being evaluated.")
    explanation: RelevanceExplanation = Field(..., description="Detailed explanation of the document relevance to the query.")
    relevance_rating: int = Field(..., description="A relevance rating on a scale from 1 to 5.")


class LLM_Evaluation:
    system_prompt = """
        You are an expert evaluator tasked with assessing the relevance of machine learning assets (such as models or datasets) to specific user queries. Each query describes the requirements and desired properties of an asset. You will be given a query and a corresponding document (a short asset description together with additional metadata) and are asked to provide the following:

        1) A detailed explanation structured into the following sections:

            - Condition Evaluations: For each key condition or constraint mentioned in the query, provide:
                - Condition: The specific condition being evaluated.
                - Met: Whether the condition was met (true/false/cant tell).
                - Details: A brief description of how the condition was or was not met.
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
    def _build_chain(
        cls, llm: BaseLLM, 
        use_openai_function_calling: bool, 
        pydantic_model: Type[BaseModel]
    ) -> SimpleChain:
        postprocess_lambda = lambda out: out[0]
        prompt_templates = [cls.system_prompt, cls.prompt_template]
    
        if use_openai_function_calling is False:
            prompt_templates[1] += "\n\n{format}"
            postprocess_lambda = None

        chain_output_opts = ChainOutputOpts(
            langchain_parser=JsonOutputParser(pydantic_object=pydantic_model),
            prompt_placeholder_name="format",
            openai_funcion_calling_schema_class=(
                pydantic_model if use_openai_function_calling else None
            )
        )
        chain_wrapper = SimpleChain(
            llm, prompt_templates, 
            chain_output_opts=chain_output_opts, 
            postprocess_lambda=postprocess_lambda
        )
        return chain_wrapper


if __name__ == "__main__":
    query = "I need a dataset containing annotated images of cats and dogs, each image having resolution of at least 1080. The dataset cannot be larger than 5GB"
    document = "Dataset XYZ contains 12,000 annotated images of various animals, including 8,000 images of cats and 4,000 images of dogs."

    llm = Ollama(model="mistral", num_predict=1024)
    use_openai_function_calling = False

    chain = LLM_Evaluation._build_chain(llm, use_openai_function_calling=False, pydantic_model=RelevanceEvaluation)
    
    output = chain.invoke({
        "query": query,
        "document": document
    })

    import json
    with open("./test.json", "w") as f:
        json.dump(output, f, ensure_ascii=False)
