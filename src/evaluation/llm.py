import os
from typing import Type
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_community.llms import Ollama
from langchain_openai.chat_models.base import BaseChatOpenAI

from lang_chains import ChainOutputOpts, SimpleChain


class LLM_Chain:
    @staticmethod
    def build_simple_chain(
        llm: BaseLLM,
        pydantic_model: Type[BaseModel],
        prompt_templates: tuple[str, str],
    ) -> SimpleChain:        
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
    

def load_llm(ollama_name: str | None = None) -> BaseLLM:
    if ollama_name is not None:
        return Ollama(model=ollama_name, num_predict=1024)
    
    azure_environs = [
        "OPENAI_API_VERSION", "AZURE_OPENAI_ENDPOINT", 
        "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT"
    ]
    for env in azure_environs:
        if os.environ.get(env, None) is None:
            break
    else:
        return AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"]
        )
    if os.environ.get("OPENAI_API_KEY", None) is not None:
        return ChatOpenAI(model="gpt-4o")
    
    return Ollama(model="mistral", num_predict=1024)