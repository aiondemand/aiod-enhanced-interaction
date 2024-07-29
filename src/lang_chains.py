import os
from tqdm import tqdm
from operator import itemgetter
from typing import Any, Type, Callable
from abc import abstractmethod, ABC
import json

from langchain.prompts.few_shot import FewShotChatMessagePromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    JsonOutputParser, StrOutputParser, BaseOutputParser
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.language_models.llms import BaseLLM
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnableLambda


class ChainOutputOpts:
    """
    Class containing relevant attributes that further process the model outuput.
    In this class we define what parser we use, whether its a JSON parser and 
    whether we use a special funcion calling tools from OpenAI that has a bit 
    different interface compared to traditional LLMs
    """
    def __init__(
        self, langchain_parser_class: Type[BaseOutputParser] | None = None,
        pydantic_model: Type[BaseModel] | None = None,
        schema_placeholder_name: str | None = None,
        use_openai_bind_tools: bool = False
    ) -> None:
        self.pydantic_model = pydantic_model
        self.schema_placeholder_name = schema_placeholder_name
        self.use_openai_bind_tools = use_openai_bind_tools

        # we want to output JSON (perform function calling)
        if langchain_parser_class is JsonOutputParser or use_openai_bind_tools:
            # we utilize OpenAI bind tools function
            if use_openai_bind_tools:
                if pydantic_model is None:
                    raise ValueError("You need to define Pydantic model that will be adhered to using OpenAI 'bind_tools' function")
                if langchain_parser_class is not None:
                    print("Warning: 'langchain_parser_class' argument is being ignored since we shall use OpenAI 'bind_tools' function instead")
                self.langchain_parser = None
            
            # we use a simple JsonOutputParser
            else:
                if pydantic_model is None:
                    raise ValueError("You need to define Pydantic model that will be adhered to using JsonOutputParser")
                if schema_placeholder_name is None:
                    raise ValueError("You need to define Schema placeholder name for JSON schema utilized in model prompt")
                self.langchain_parser = langchain_parser_class(pydantic_model=pydantic_model)
        
        # we utilize a different parser 
        else:
            if langchain_parser_class is None:
                langchain_parser_class = StrOutputParser()
            self.langchain_parser = langchain_parser_class()

    def augment_prompt_with_json_schema(
        self, prompt: ChatPromptTemplate
    ) -> ChatPromptTemplate:
        if isinstance(self.langchain_parser, JsonOutputParser) is False:
            return prompt

        schema = self.langchain_parser.get_format_instructions()
        return prompt.partial(**{
            self.schema_placeholder_name: schema
        })
    
    def augment_llm_with_openai_function_calling_tool(
        self, llm: BaseLLM
    ) -> tuple[BaseLLM, BaseOutputParser]:
        if self.use_openai_bind_tools is False:
            return llm, self.langchain_parser
        
        schema_name = self.pydantic_model.__name__
        llm_fc = llm.bind_tools([self.pydantic_model], tool_choice=schema_name)
        return llm_fc, JsonOutputKeyToolsParser(key_name=schema_name)
    

class Chain(ABC):
    """
    Generic class representing logic of building a Langchain chain

    We can define the prompt templates, few shot setting and output processing
    defined by the ChainOutputOpts object. Also the 'postprocess_lambda' argument
    defines the very last step in the chain that should be performed to extract 
    the information we seek to retrieve
    """
    def __init__(
        self, llm: BaseLLM, 
        prompt_templates: tuple[str, str],
        fewshot_examples: list[dict[str, str]] | None = None, 
        fewshot_prompt_templates: str | None = None,
        chain_output_opts: ChainOutputOpts | None = None,
        postprocess_lambda: Callable[[dict | str], dict | str] | None = None
    ) -> None:
        self.llm = llm
        self.prompt_templates = prompt_templates
    
        if fewshot_examples is None != fewshot_prompt_templates is None:
            raise ValueError(
                "You need to define both the few-shot examples, " + 
                "as well as their corresponding prompt templates"
            )
        self.fewshot_examples = fewshot_examples
        self.fewshot_prompt_templates = fewshot_prompt_templates
        
        self.chain_output_opts = chain_output_opts
        if self.chain_output_opts is None:
            self.chain_output_opts = ChainOutputOpts()

        self.postprocess_lambda = postprocess_lambda
        if self.postprocess_lambda is None:
            self.postprocess_lambda = lambda out: out

        self.prompt = build_prompt(
            self.prompt_templates,
            self.fewshot_examples, 
            self.fewshot_prompt_templates,
            self.chain_output_opts
        )
        
    @abstractmethod
    def build_chain(self) -> RunnableSequence:
        pass

    @abstractmethod
    def invoke(self, *args, **kwargs) -> Any:
        pass


class TwoStageChain(Chain):
    """
    A class representing Langchain chain that has two stages
    
    The first stage usually performs the main task 
    while the second one is used for purposes such as 
    self-reflection, explicit information extraction from previous model
    response, ...
    """
    def __init__(
        self, llm: BaseLLM, 
        prompt_template: tuple[str, str],
        second_prompt_template: str,
        fewshot_examples: list[dict[str, str]] | None = None, 
        fewshot_prompt_templates: str | None = None,
        chain_output_opts: ChainOutputOpts | None = None,
        postprocess_lambda: Callable[[dict | str], dict | str] | None = None
    ) -> None:
        super().__init__(
            llm, prompt_template, fewshot_examples, 
            fewshot_prompt_templates, chain_output_opts,
            postprocess_lambda
        )
        self.second_prompt_template = second_prompt_template
        self.chain = self.build_chain()

    def build_chain(self) -> RunnableSequence:        
        second_stage_prompt = ChatPromptTemplate.from_messages([
            ("ai", "{model_response}"),
            ("human", self.second_prompt_template)
        ])
        second_stage_prompt = self.chain_output_opts.augment_prompt_with_json_schema(
            self.prompt + second_stage_prompt
        )

        llm2, parser = \
            self.chain_output_opts.augment_llm_with_openai_function_calling_tool(
                self.llm
            )
        main_chain = self.prompt | self.llm | StrOutputParser()
        second_stage_chain = second_stage_prompt | llm2 | parser

        entire_chain = (
            RunnableParallel({
                "model_response": main_chain,
                "doc": itemgetter("doc")
            }) |
            second_stage_chain |
            RunnableLambda(self.postprocess_lambda)
        )
        return entire_chain

    def invoke(self, input: dict, num_retry_attempts: int = 3) -> dict | str | None:
        for _ in range(num_retry_attempts):
            try:
                pred = self.chain.invoke(input)
                self.chain_output_opts.pydantic_model(**pred)
                return pred
            except:
                continue
        return None
    
    
class SimpleChain(Chain):
    """
    A class representing a simple chain consisting of a prompt, an LLM and a parser
    """
    def __init__(
        self, llm: BaseLLM, 
        prompt_templates: tuple[str, str],
        fewshot_examples: list[dict[str, str]] | None = None, 
        fewshot_prompt_templates: str | None = None,
        chain_output_opts: ChainOutputOpts | None = None,
        postprocess_lambda: Callable[[dict | str], dict | str] = None
    ) -> None:
        super().__init__(
            llm, prompt_templates, fewshot_examples, 
            fewshot_prompt_templates, chain_output_opts,
            postprocess_lambda
        )
        self.chain = self.build_chain()

    def build_chain(self) -> RunnableSequence:
        llm, parser = \
            self.chain_output_opts.augment_llm_with_openai_function_calling_tool(
                self.llm
            )
        return self.prompt | llm | parser | RunnableLambda(self.postprocess_lambda)

    def invoke(self, input: dict, num_retry_attempts: int = 3) -> dict | str | None:
        for _ in range(num_retry_attempts):
            try:
                pred = self.chain.invoke(input)
                self.chain_output_opts.pydantic_model(**pred)
                return pred
            except:
                continue
        return None


def build_prompt(
    prompt_templates: tuple[str, str],
    fewshot_examples: list[dict[str, str]], 
    fewshot_prompt_templates: tuple[str],
    chain_output_opts: ChainOutputOpts
) -> FewShotChatMessagePromptTemplate | ChatPromptTemplate:
    """
    Function for building initial prompts for Langchain chains

    This function also takes into consideration the few-shot setting 
    when building the prompt that is further fed into the model
    """
    fewshot_prompt = None
    if fewshot_examples is not None and fewshot_prompt_templates is not None:
        human_prompt_template, ai_prompt_template = fewshot_prompt_templates
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", human_prompt_template),
            ("ai", ai_prompt_template),
        ])

        fewshot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=fewshot_examples,
        )

    system_prompt_template, human_prompt_template = prompt_templates
    chat_messages = [
        ("system", system_prompt_template),
        ("human", human_prompt_template)
    ]
    if fewshot_prompt is not None:
        chat_messages = [
            ("system", system_prompt_template),
            fewshot_prompt,
            ("human", human_prompt_template)
        ]
    
    prompt = ChatPromptTemplate.from_messages(chat_messages) 
    return chain_output_opts.augment_prompt_with_json_schema(prompt)


# TODO make it work with dataLoader perhaps -> might be more accessible then...
def apply_chains_on_files_in_directory(
    primary_chain: Chain, 
    dirpath: str, savedir: str,
    format_input_fn: Callable[[str], dict],
    backup_chain: Chain | None = None,
    verbose: bool = True, num_attempts: int = 3,
    check_validity_fn: Callable[[dict | str], bool] | None = None,
) -> list[str]:
    chains_to_use = [primary_chain]
    if backup_chain is not None:
        chains_to_use = [primary_chain, backup_chain]

    os.makedirs(savedir, exist_ok=True)
    failed_docs = []
    for file in tqdm(sorted(os.listdir(dirpath)), disable=verbose is False):
        filename_no_ext = file[:file.rfind(".")]
        savepath = os.path.join(savedir, f"{filename_no_ext}.json")
        if os.path.exists(savepath):
            continue
        with open(os.path.join(dirpath, file), encoding="utf-8") as f:
            text = f.read()

        success = False
        for chain in chains_to_use:
            for _ in range(num_attempts):
                try:
                    response = chain.invoke(format_input_fn(text))
                    is_valid_response = check_validity_fn(response)
                    if is_valid_response is False:
                        raise ValueError("Invalid response format")
                except:
                    continue
                
                with open(savepath, "w", encoding="utf-8") as f:
                    json.dump(response, f, ensure_ascii=False)
                success = True
                break
            if success:
                break

        if success is False:
            failed_docs.append(file)
    return failed_docs