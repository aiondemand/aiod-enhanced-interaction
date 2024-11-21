from ast import literal_eval
from operator import itemgetter
import os
import json
import re
from typing import Any, Self, Literal, Union, Optional, Type, TypeVar, get_origin, get_args, TypeAlias
from enum import Enum
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langchain_community.callbacks import get_openai_callback
from langchain_core.output_parsers import (
    JsonOutputParser, StrOutputParser, BaseOutputParser
)
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.language_models.llms import BaseLLM
from langchain_core.runnables import RunnableLambda
from torch.utils.data import DataLoader

from dataset import Queries
from lang_chains import ChainOutputOpts, LLM_Chain, SimpleChain, load_llm
from data_types import AnnotatedDoc, QueryDatapoint, SemanticSearchResult


def apply_lowercase(obj: Any) -> Any:
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = apply_lowercase(v)
    elif isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = apply_lowercase(obj[i])
    elif isinstance(obj, str):
        obj = obj.lower()

    return obj


class Condition(BaseModel):
    """Condition pertaining to a specific metadata field we can use to filter out ML assets in database"""
    values: list[Union[str, int, float]] = Field(..., description=f"The values associated with the condition applied to a specific metadata field. Each value is evaluated against the metadata field separately. The values have the same data type and format restrictions imposed to them as the metadata field itself."),
    comparison_operator: Literal["<", ">", "<=", ">=", "==", "!="] = Field(..., description="The comparison operator that determines how all the values should be compared to the metadata field."),
    logical_operator: Literal["AND", "OR"] = Field(..., description="The logical operator that performs logical operations (AND/OR) in between multiple expressions corresponding to each extracted value. If there's only one extracted value pertaining to this metadata field, set this attribute to AND."),
            

############################################################
############################################################
############################################################
############################################################
############################################################

class DatasetMetadataTemplate(BaseModel):
    """
    Extraction of relevant metadata we wish to retrieve from ML assets
    """
    
    platform: Literal["huggingface", "openml", "zenodo"] = Field(
        ..., 
        description="The platform where the asset is hosted. ONLY PERMITTED VALUES: ['huggingface', 'openml', 'zenodo']"
    )
    date_published: str = Field(
        ..., 
        description="The original publication date of the asset in the format 'YYYY-MM-DD'."
    )
    year: int = Field(
        ..., 
        description="The year extracted from the publication date."
    )
    month: int = Field(
        ..., 
        description="The month extracted from the publication date."
    )
    domains: Optional[list[Literal["NLP", "Computer Vision", "Audio Processing"]]] = Field(
        None, 
        description="The AI technical domains of the asset, describing the type of data and AI task involved. ONLY PERMITTED VALUES: ['NLP', 'Computer Vision', 'Audio Processing']. Leave the list empty if not specified."
    )
    task_types: Optional[list[str]] = Field(
        None, 
        description="The machine learning tasks supported by this asset. Acceptable values include task types found on HuggingFace (e.g., 'token-classification', 'question-answering', ...). Leave the list empty if not specified"
    )
    license: Optional[str] = Field(
        None, 
        description="The license type governing the asset usage, if specified."
    )

    # Dataset-specific metadata
    size_in_mb: Optional[float] = Field(
        None, 
        description="The total size of the dataset in megabytes. If the size is not explicitly specified in the dataset descritpion, sum up the sizes of individual files instead if possible. Don't forget to convert the sizes to MBs"
    )
    num_datapoints: Optional[int] = Field(
        None, 
        description="The number of data points in the dataset, if specified."
    )
    size_category: Optional[str] = Field(
        None, 
        description="The general size category of the dataset, typically specified in ranges such as '10k<n<100k' found on HuggingFace. If you know the precise number of datapoints you may infer the size category."
    )
    modalities: Optional[list[Literal["text", "tabular", "audio", "video", "image"]]] = Field(
        None, 
        description="The modalities present in the dataset, such as 'text', 'tabular', 'audio', 'video', or 'image'. Leave the list empty if not specified"
    )
    data_formats: Optional[list[str]] = Field(
        None, 
        description="The file formats of the dataset (e.g., 'CSV', 'JSON', 'Parquet'), if specified. Leave the list empty if not specified"
    )
    languages: Optional[list[str]] = Field(
        None, 
        description="Languages present in the dataset, specified in ISO 639-1 two-letter codes (e.g., 'EN' for English, 'ES' for Spanish, 'FR' for French, ...). Leave the list empty if not specified"
    )


def is_optional_type(annotation: Type) -> bool:
    if get_origin(annotation) is Union:
        return type(None) in get_args(annotation)
    return False


def strip_optional_type(annotation: Type) -> Type:
    if get_origin(annotation) is Union:
        args = get_args(annotation)
        if type(None) in args:
            return next(arg for arg in args if arg is not type(None))
    return annotation


def wrap_in_list_type_if_not_already(annotation: Type) -> Type:
    if get_origin(annotation) is not list:
        annotation = list[annotation]
    return annotation


def user_query_metadata_extraction_schema_factory(
    template_type: Type[BaseModel], 
    simplified_schema: bool = False
) -> Type[BaseModel]:
    schema_type_name = f'UserQuery_MetadataSchema{"_Simplified" if simplified_schema else ""}_{template_type.__name__}'

    if simplified_schema is False:
        new_inner_value_annotations = {
            k: wrap_in_list_type_if_not_already(strip_optional_type(v))
            for k, v in template_type.__annotations__.items()
        }

        descr_template = "Conditions pertaining to the metadata field `{field_name}:` {description}"
        new_field_values = {
            k: Field(None, description=descr_template.format(field_name=k, description=v.description))
            for k, v in template_type.model_fields.items()
        }
        new_field_annotations = {
            k: user_query_field_factory(
                field_type_name=f"{schema_type_name}_{k}",
                data_type=v
            )
            for k, v in new_inner_value_annotations.items()
        }

        arguments = {
            '__annotations__': new_field_annotations,
            "__doc__": "Extraction of user-defined conditions on metadata fields we can filter by"
        }
        arguments.update(new_field_values)
        return type(schema_type_name, (BaseModel, ), arguments)
    
    # Simplified schema



def user_query_field_factory(
    field_type_name: str, 
    data_type: Type[BaseModel], 
) -> Type[BaseModel]:    
    return Optional[list[type(
        field_type_name,  
        (BaseModel, ),
        {
            '__annotations__': {
                "values": data_type,
                "comparison_operator": Literal["<", ">", "<=", ">=", "==", "!="],
                "logical_operator": Literal["AND", "OR"]
            },
            'values': Field(..., description=f"The values associated with the condition (or more precisely with expressions constituting the entirery of the condition) applied to specific metadata field."),
            'comparison_operator': Field(..., description="The comparison operator that determines how the values should be compared to the metadata field in their respective expresions."),
            'logical_operator': Field(..., description="The logical operator that performs logical operations (AND/OR) in between multiple expressions corresponding to each extracted value. If there's only one extracted value pertaining to this metadata field, set this attribute to AND."),
        }
    )]]


def build_milvus_filter(data: dict) -> str:    
    simple_expression_template = "({field} {op} {val})"
    list_expression_template = "({op}ARRAY_CONTAINS({field}, {val}))"
    format_value = lambda x: f"'{x.lower()}'" if isinstance(x, str) else x
    list_fields = {
        k: get_origin(strip_optional_type(v)) is list
        for k, v in DatasetMetadataTemplate.__annotations__.items()
    }

    condition_strings = []
    for field_name, conditions in data.items():
        for condition in conditions:    
            comp_operator = condition["comparison_operator"]
            log_operator = condition["logical_operator"]
            expressions = []
            for value in condition["values"]:
                if list_fields[field_name]:
                    if comp_operator not in ["==", "!="]:
                        raise ValueError("We don't support any other operators but a check whether values exist whithin the list in the asset.")
                    expressions.append(
                        list_expression_template.format(
                            field=field_name,
                            op="" if comp_operator == "==" else "not ",
                            val=format_value(value)
                        )
                    )
                else:
                    expressions.append(
                        simple_expression_template.format(
                            field=field_name,
                            op=comp_operator,
                            val=format_value(value)
                        )
                    )

            condition_strings.append(
                " ".join(expressions)
                if len(expressions) < 2
                else "(" + f" {log_operator.lower()} ".join(expressions) + ")"
            )

    return " and ".join(condition_strings)
            

class Llama_ManualFunctionCalling:
    tool_prompt_template = """
        You have access to the following functions:

        Use the function '{function_name}' to '{function_description}':
        {function_schema}

        If you choose to call a function ONLY reply in the following format with no prefix or suffix:

        <function=example_function_name>{{\"example_name\": \"example_value\"}}</function>

        Reminder:
        - Function calls MUST follow the specified format, start with <function= and end with </function>
        - Required parameters MUST be specified
        - Only call one function at a time
        - Put the entire function call reply on one line
        - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
    """

    def __init__(
        self, llm: ChatOllama, pydantic_model: Type[BaseModel], 
        user_query_text: str, examples: list[dict] | None = None
    ) -> None:
        self.llm = llm
        self.pydantic_model = pydantic_model
        self.user_query_template = user_query_text
        self.examples = examples

        # build prompt
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "User Query: {input}"),
            ("ai", "{output}"),
        ])
        fewshot_prompt = FewShotChatMessagePromptTemplate(
            examples=self.transform_fewshot_examples(examples),
            example_prompt=example_prompt
        )
        composite_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(self.populate_tool_prompt(pydantic_model)), # no input variables, hence the use of message
            HumanMessage(user_query_text), # no input variables, hence the use of message
            fewshot_prompt,
            ("human", "User Query: {query}"),
        ])

        self.chain = composite_prompt | self.llm | StrOutputParser() | RunnableLambda(
            self.convert_llm_string_output_to_tool
        )

    def __call__(self, query: str) -> dict | None:
        out = self.chain.invoke({"query": query})

        try:
            self.pydantic_model(**out)
        except Exception as e:
            print(e)
            return None
        return out
    
    def transform_fewshot_examples(self, examples: list[dict]) -> str:
        return [
            {
              "input": ex["input"],
              "output": f"<function={self.pydantic_model.__name__}>{json.dumps(ex['output'])}</function>"
            } for ex in examples
        ]
    
    def convert_llm_string_output_to_tool(self, response: str) -> dict | None:
        function_regex = r"<function=(\w+)>(.*?)</function>"
        match = re.search(function_regex, response)

        if match:
            function_name, args_string = match.groups()
            try:
                return literal_eval(args_string)
            except:
                print(f"Error parsing function arguments")
                return None
        return None

    def populate_tool_prompt(self, pydantic_model: Type[BaseModel]) -> str:
        tool_schema = self.transform_simple_pydantic_schema_to_tool_schema()

        return self.tool_prompt_template.format(
            function_name=tool_schema["name"],
            function_description=tool_schema["description"],
            function_schema=json.dumps(tool_schema)
        )

    def transform_simple_pydantic_schema_to_tool_schema(self) -> dict:
        pydantic_schema = self.pydantic_model.model_json_schema()
        
        pydantic_schema.pop("type")
        pydantic_schema["name"] = pydantic_schema.pop("title")
        pydantic_schema["parameters"] = {
            "type": "object",
            "properties": pydantic_schema.pop("properties"),
            "required": pydantic_schema.pop("required")
        }

        return pydantic_schema


class LLM_MetadataExtractor:
    system_prompt_from_asset = """
        ### Task Overview:
        You are tasked with extracting specific metadata attributes from a document describing a machine learning {asset_type}. This metadata will be used for database filtering, so precision, clarity, and high confidence are essential. The extraction should focus only on values you are highly confident in, ensuring they are concise, unambiguous, and directly relevant.

        ### Instructions:
        - Extract Only High-Confidence Metadata: Only extract values if you are certain of their accuracy based on the provided document. If you are unsure of a particular metadata value, skip the attribute.
        - Align Values with Controlled Terminology: Some extracted values may need minor modifications to match acceptable or standardized terminology (e.g., specific task types or domains). Ensure that values align with controlled terminology, making them consistent and unambiguous.
        - Structured and Minimal Output: The extracted metadata must align strictly with the provided schema. Avoid adding extra information or interpretations that are not directly stated in the document. If multiple values are relevant for a field, list them as specified by the schema.
        - Focus on Database Usability: The extracted metadata will be used to categorize and filter {asset_type}s in a database, so each value should be directly useful for this purpose. This means avoiding ambiguous or verbose values that would complicate search and filtering.
    """
    system_prompt_from_user_query = """
        ### Task Overview:
        You are an advanced language model skilled in extracting structured conditions from natural language user queries used to filter and search for specific {asset_type}s in the database. 
        Your task is to identify conditions for predefined set of metadata fields that are a part of JSON schema defined at the end of this prompt representing the form your output needs to adhere to.
        There may be zero, one or multiple conditions correspoding to each metadata field.

        Each condition must include three following components:
        1. **Comparison Operator**: The comparison operator that determines how the values should be compared to the metadata field in their respective expresions (e.g., `==`, `>`, `<`, `!=`, ...).
        2. **Logical operator**: The logical operator that performs either logical `AND` or logical `OR` in between potentionally multiple expressions corresponding to each extracted value.
        This field is only meaningful if there are multiple values associated with this condition, otherwise set this value to `AND`.
        3. **Values**: The values associated with the condition applied to specific metadata field. There may be multiple values, each of them creating their own expression comparing themselves to the metadata field 
        utilizing the same comparison operator and then the logical operator is applied in between all the expressions to connect the expressions into one condition.
            
        If a condition cannot be fully constructed (i.e., is missing either the comparison operator, the logical operator or the values field), do not include it in the output. 
        
        Some extracted values may need minor modifications to match acceptable or standardized terminology or formattign. Ensure that values align with controlled terminology, making them consistent and unambiguous.
        The types of requested values correspoding to the individual metadata fields are defined in the JSON schema below.
    """

    user_query_extraction_examples_hierarchical = """
        ### Examples of the task:
        **User query:** "Find all chocolate datasets created after January 1, 2022, that are represented in textual or image format with its dataset size smaller than 500 000KB."
 
        **Output:**
        {{
            "date_published": [
                {{
                    "values": ["2022-01-01T00:00:00"],
                    "comparison_operator": ">=",
                    "logical_operator": "AND"
                }}
            ],
            "modalities": [
                {{
                    "values": ["text","image"],
                    "comparison_operator": "==",
                    "logical_operator": "OR",
                }}
            ],
            "size_in_mb": [
                {{
                    "values": [488],
                    "comparison_operator": "<",
                    "logical_operator": "AND"
                }}
            ]
        }}

        **User query:** "Show me the multilingual summarization datasets containing both the French as well as English data. The dataset however can't include any German data nor any Slovak data."
 
        **Output:**
        {{
            "task_types": [
                {{
                    "values": ["summarization"],
                    "comparison_operator": "==",
                    "logical_operator": "AND"
                }}
            ],
            "languages": [
                {{
                    "values": ["fr", "en"],
                    "comparison_operator": "==",
                    "logical_operator": "AND"
                }},
                {{
                    "values": ["de", "sk"],
                    "comparison_operator": "!=",
                    "logical_operator": "AND"
                }},
            ]
        }}
    """
    user_query_extraction_examples_flatten = """
        ### Examples of the task:
        **User query:** "Find all chocolate datasets created after January 1, 2022, that are represented in textual or image format with its dataset size smaller than 500 000KB."
 
        **Output:**
        {{
            "date_published__values": [["2022-01-01T00:00:00"]],
            "date_published__comparison_operator": [">="],
            "date_published__logical_operator": ["AND"]  
            "modalities__values": [["text", "image"]],
            "modalities__comparison_operator": ["=="],
            "modalities__logical_operator": ["OR"],
            "size_in_mb__values": [[488]],
            "size_in_mb__comparison_operator": ["<"],
            "size_in_mb__logical_operator": ["AND"],
        }}

        **User query:** "Show me the multilingual summarization datasets containing both the French as well as English data. The dataset however can't include any German data or any Slovak data."
 
        **Output:**
        {{
            "task_types__values": [["summarization"]],
            "task_types__comparison_operator": ["=="],
            "task_types__logical_operator": ["AND"],
            "languages__values": [["fr", "en"], ["de", "sk"]],
            "languages__comparison_operator": ["==", "!="],
            "languages__logical_operator": ["AND", "AND"]
        }}
    """

    user_prompt_from_asset = """
        ### Document of machine learning {asset_type} to extract metadata from based on the provided schema:
        {document}
    """
    user_prompt_from_user_query = """
        ### User query containing potential conditions to extract and filter machine learning {asset_type}s on based on the provided schema:
        {query}
    """

    @classmethod
    def build_chain(
        cls, 
        llm: BaseLLM | None = None,
        pydantic_model: Type[BaseModel] | None = None,
        asset_type: Literal["dataset", "model"] = "dataset",
        parsing_user_query: bool = False,
    ) -> SimpleChain:
        if llm is None:
            llm = load_llm()

        if pydantic_model is None:
            pydantic_model = (
                user_query_metadata_extraction_schema_factory(
                     template_type=DatasetMetadataTemplate
                ) 
                if parsing_user_query
                else DatasetMetadataTemplate
            )
        
        system_prompt = (
            cls.system_prompt_from_user_query 
            if parsing_user_query 
            else cls.system_prompt_from_asset
        )
        examples_prompt = (
            "" if parsing_user_query is False else
            cls.user_query_extraction_examples_hierarchical
        )
        system_prompt = system_prompt.format(asset_type=asset_type) + examples_prompt
        user_prompt = (
            cls.user_prompt_from_user_query 
            if parsing_user_query 
            else cls.user_prompt_from_asset
        ) 
        prompt_templates = [
            system_prompt,
            user_prompt
        ]

        return LLM_Chain.build_simple_chain(llm, pydantic_model, prompt_templates)
    
    def __init__(
        self, chain: SimpleChain | None = None, 
        asset_type: Literal["dataset", "model"] = "dataset",
        parsing_user_query: bool = False,
    ) -> None:
        self.asset_type = asset_type
        self.extracting_from_user_query = parsing_user_query
        
        self.chain = chain
        if chain is None:
            self.chain = self.build_chain(
                asset_type=asset_type,
                parsing_user_query=parsing_user_query
            )

    def __call__(self, doc: str) -> dict | str | None:
        input = {
            "query" if self.extracting_from_user_query else "document": doc,
            "asset_type": self.asset_type
        }
        return self.chain.invoke(input)


class NaturalLanguageCondition(BaseModel):
    condition: str = Field(
        ..., 
        description="Natural language condition corresponding to a particular metadata field we use for filtering. It may contain either only one value to be compared to metadata field, or multiple values if there's an OR logical operator in between those values"
    )
    field: str = Field(..., description="Name of the metadata field")
    operator: Literal["AND", "OR", "NONE"] = Field("NONE", description="Logical operator used between multiple values pertaining to the same metadata field. If the condition describes only one value, set it to NONE instead.")


class UserQueryParsing(BaseModel):
    """Extraction and parsing of conditions and a topic found within a user query"""
    topic: str = Field(..., description="A topic or a main subject of the user query that user seeks for")
    conditions: list[NaturalLanguageCondition] = Field(..., description="Natural language conditions")


if __name__ == "__main__":
    MODEL_NAME = "llama3.1:8b"

    attribute_types = {
        "platform": "string",
        "date_published": "string",
        "year": "integer",
        "month": "integer",
        "domains": "string",
        "task_types": "string",
        "license": "string",
        "size_in_mb": "float",
        "num_datapoints": "integer",
        "size_category": "string",
        "modalities": "string",
        "data_formats": "string",
        "languages": "string",
    }
    metadata_field_info = [
        {
            "name": name, 
            "description": field.description, 
            "type": attribute_types[name]
        } for name, field in DatasetMetadataTemplate.model_fields.items()
    ]

    path = "src/fewshot_examples/user_query_extraction_stage1.json"
    with open(path) as f:
        fewshot_examples = json.load(f)    

    user_prompt = """
        Your task is to process a user query that may contain multiple natural language conditions and a general topic. Each condition corresponds to a specific metadata field and describes one or more values that should be compared against that field.
        These conditions are subsequently used to filter out unsatisfactory data in database. On the other hand, the topic is used in semantic search to find the most relevant documents to the thing user seeks for
        
        A simple schema below briefly describes all the metadata fields we use for filtering purposes:
        {model_schema}

        **Key Guidelines:**

        1. **Conditions and Metadata Fields:**
        - Each condition must clearly correspond to exactly one metadata field that we use for filtering purposes.
        - If a condition references multiple metadata fields (e.g., "published after 2020 or in image format"), split it into separate conditions with NONE operators, each tied to its respective metadata field.

        2. **Handling Multiple Values:**
        - If a condition references multiple values for a single metadata field (e.g., "dataset containing French or English"), include all the values in the natural language condition.
        - Specify the logical operator (AND/OR) that ties the values:
            - Use **AND** when the query requires all the values to match simultaneously.
            - Use **OR** when the query allows any of the values to match.

        3. **Natural Language Representation:**
        - Preserve the natural language form of the conditions. You're also allowed to modify them slightly to maintain their meaning once they're extracted from their context

        4. **Logical Operators for Conditions:**
        - Always include a logical operator (AND/OR) for conditions with multiple values.
        - For conditions with a single value, the logical operator is not required but can default to "NONE" for clarity.

        5. **Extract user query topic**
        - A topic is a concise, high-level description of the main subject of the query. 
        - It should exclude specific filtering conditions but capture the core concept or intent of the query.
        - If the query does not explicitly state a topic, infer it based on the overall context of the query.

    """

    user_prompt = user_prompt.format(model_schema=json.dumps(metadata_field_info))

    model = ChatOllama(model=MODEL_NAME, num_predict=4096, num_ctx=8192)

    extraction_stage1 = Llama_ManualFunctionCalling(
        model, 
        UserQueryParsing, 
        user_query_text=user_prompt, 
        examples=fewshot_examples
    )

    user_query = (
        "Retrieve all the translation Stanford datasets that have more than 10k datapoints and fewer than 100k datapoints. The dataset has over 100k KB in size " +
        "and the dataset should contain one or more of the following languages: Slovak, Polish, French, German. However we don't wish the dataset to contain any English nor Spanish"
    )
    # user_query = "I dont want news datasets with any textual or video data."
    # user_query = "I want datasets with Slovak or English. It also needs to have either French or German"


    out = extraction_stage1(user_query)


    exit()

    # from preprocess.text_operations import ConvertJsonToString
    # with open("temp/data_examples/huggingface.json") as f:
    #     data = json.load(f)[0]
    # text_format = ConvertJsonToString().extract_relevant_info(data)
    
    # llm_chain = LLM_MetadataExtractor.build_chain(llm=load_llm(ollama_name=MODEL_NAME), parsing_user_query=False)
    # extract = LLM_MetadataExtractor(
    #     chain=llm_chain,
    #     asset_type="dataset", 
    #     parsing_user_query=False,
    # )

    # out = extract(text_format)

    # exit()

    # user_query = (
    #     "Retrieve all the summarization datasets with at least 10k datapoints, yet no more than 100k datapoints, " +
    #     "and the dataset should have contain Slovak language, Polish language, but no Czech language."
    # )
    # user_query_2 = (
    #     "Retrieve all translation datasets that either have at least 10k datapoints and has over 100k KB in size" +
    #     "or they contain Slovak language and Polish language, but no Czech language."
    # )
    
    # llm_chain = LLM_MetadataExtractor.build_chain(llm=load_llm(ollama_name=MODEL_NAME), parsing_user_query=True)
    # extract_query = LLM_MetadataExtractor(
    #     chain=llm_chain, 
    #     asset_type="dataset", 
    #     parsing_user_query=True
    # )

    # output = extract_query(user_query_2)
    # build_milvus_filter(output)


