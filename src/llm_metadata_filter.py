from copy import deepcopy
from functools import partial
from ast import literal_eval
from operator import itemgetter
import os
import json
import re
from typing import Any, Callable, Self, Literal, Union, Optional, Type, TypeVar, get_origin, get_args, TypeAlias
from enum import Enum
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langchain_community.callbacks import get_openai_callback
from langchain_core.output_parsers import (
    JsonOutputParser, StrOutputParser, BaseOutputParser
)
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, PromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.language_models.llms import BaseLLM
from langchain_core.runnables import RunnableLambda, RunnableSequence
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


# Classes pertaining to the first stage of user query parsing
class NaturalLanguageCondition(BaseModel):
    condition: str = Field(
        ..., 
        description="Natural language condition corresponding to a particular metadata field we use for filtering. It may contain either only one value to be compared to metadata field, or multiple values if there's an OR logical operator in between those values"
    )
    field: str = Field(..., description="Name of the metadata field")
    
    # helper field used for better operator analysis. Even though the model doesnt assign operator correctly all the time
    # it's a way of forcing the model to focus on logical operators in between conditions
    # since the value of this attribute is unreliable we dont use it in the second stage
    operator: Literal["AND", "OR", "NONE"] = Field(
        ..., 
        description="Logical operator used between multiple values pertaining to the same metadata field. If the condition describes only one value, set it to NONE instead."
    )


class SurfaceQueryParsing(BaseModel):
    """Extraction and parsing of conditions and a topic found within a user query"""
    topic: str = Field(..., description="A topic or a main subject of the user query that user seeks for")
    conditions: list[NaturalLanguageCondition] = Field(..., description="Natural language conditions")
            

# Classes representing specific asset metadata 
class DatasetMetadataTemplate(BaseModel):
    """
    Extraction of relevant metadata we wish to retrieve from ML assets
    """
    
    platform: Literal["huggingface", "openml", "zenodo"] = Field(
        ..., 
        description="The platform where the asset is hosted. Only permitted values: ['huggingface', 'openml', 'zenodo']"
    )
    date_published: str = Field(
        ..., 
        description="The original publication date of the asset in the format 'YYYY-MM-DDTHH-MM-SS'."
    )
    year: int = Field(
        ..., 
        description="The year extracted from the publication date in integer data type."
    )
    month: int = Field(
        ..., 
        description="The month extracted from the publication date in integer data type."
    )
    domains: Optional[list[Literal["NLP", "Computer Vision", "Audio Processing"]]] = Field(
        None, 
        description="The AI technical domains of the asset, describing the type of data and AI task involved. Only permitted values: ['NLP', 'Computer Vision', 'Audio Processing']"
    )
    task_types: Optional[list[str]] = Field(
        None, 
        description="The machine learning tasks supported by this asset. Acceptable values include task types found on HuggingFace (e.g., 'token-classification', 'question-answering', ...)"
    )
    license: Optional[str] = Field(
        None, 
        description="The license type governing the asset usage, e.g., 'mit', 'apache-2.0'"
    )

    # Dataset-specific metadata
    size_in_mb: Optional[float] = Field(
        None, 
        description="The total size of the dataset in megabytes (float). If the size is not explicitly specified in the dataset descritpion, sum up the sizes of individual files instead if possible. Don't forget to convert the sizes to MBs"
    )
    num_datapoints: Optional[int] = Field(
        None, 
        description="The number of data points in the dataset in integer data type"
    )
    size_category: Optional[str] = Field(
        None, 
        description="The general size category of the dataset, typically specified in ranges such as '1k<n<10k', '10k<n<100k', etc... found on HuggingFace."
    )
    modalities: Optional[list[Literal["text", "tabular", "audio", "video", "image"]]] = Field(
        None, 
        description="The modalities present in the dataset. Only permitted values: ['text', 'tabular', 'audio', 'video', 'image']"
    )
    data_formats: Optional[list[str]] = Field(
        None, 
        description="The file formats of the dataset (e.g., 'CSV', 'JSON', 'Parquet')."
    )
    languages: Optional[list[str]] = Field(
        None, 
        description="Languages present in the dataset, specified in ISO 639-1 two-letter codes (e.g., 'en' for English, 'es' for Spanish, 'fr' for French, etc ...)."
    )


class ModelMetadataTemplate(BaseModel):
    pass


ASSET_METADATA_SCHEMAS = {
    "dataset": DatasetMetadataTemplate,
    "model": ModelMetadataTemplate
}


def is_optional_type(annotation: Type) -> bool:
    if get_origin(annotation) is Union:
        return type(None) in get_args(annotation)
    return False


def is_list_type(annotation: Type) -> bool:
    return get_origin(annotation) is list

def strip_optional_type(annotation: Type) -> Type:
    if is_optional_type(annotation):
        return next(arg for arg in get_args(annotation) if arg is not type(None))
    return annotation


def strip_list_type(annotation: Type) -> Type:
    if is_list_type(annotation):
        return get_args(annotation)[0]
    return annotation


def wrap_in_list_type_if_not_already(annotation: Type) -> Type:
    if is_list_type(annotation):
        return annotation
    return list[annotation]
    

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


def build_milvus_filter(data: dict, asset_schema: Type[BaseModel]) -> str:    
    simple_expression_template = "({field} {op} {val})"
    list_expression_template = "({op}ARRAY_CONTAINS({field}, {val}))"
    format_value = lambda x: f"'{x.lower()}'" if isinstance(x, str) else x
    list_fields = {
        k: get_origin(strip_optional_type(v)) is list
        for k, v in asset_schema.__annotations__.items()
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
        self, 
        llm: ChatOllama, 
        pydantic_model: Type[BaseModel] | None, 
        chat_prompt_no_system: ChatPromptTemplate,
        call_function: Callable[[RunnableSequence, dict], dict | None] = None,
    ) -> None:
        self.pydantic_model = pydantic_model
        self.call_function = call_function

        composite_prompt = ChatPromptTemplate.from_messages([
            ("system", "{system_prompt}"),
        ]) + chat_prompt_no_system
        if pydantic_model is not None:
            composite_prompt = composite_prompt.partial(
                system_prompt = self.populate_tool_prompt(pydantic_model)
            )

        self.chain = (
            composite_prompt | 
            llm | 
            StrOutputParser() | 
            RunnableLambda(
                self.convert_llm_string_output_to_tool
            )
        )

    def __call__(self, input: dict) -> dict | None:
        if self.call_function is not None:
            return self.call_function(self.chain, input)

        out = self.chain.invoke(input)
        if self.validate_output(out, self.pydantic_model) is False:
            return None
        return out
    

    @classmethod
    def validate_output(
        cls, output: dict, pydantic_model: Type[BaseModel] | None
    ) -> bool:
        if pydantic_model is not None:
            try:
                pydantic_model(**output)
            except Exception as e:
                return False
        return True
    
    @classmethod
    def transform_fewshot_examples(
        cls, pydantic_model: Type[BaseModel], examples: list[dict]
    ) -> str:
        return [
            {
              "input": ex["input"],
              "output": f"<function={pydantic_model.__name__}>{json.dumps(ex['output'])}</function>"
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

    @classmethod
    def populate_tool_prompt(cls, pydantic_model: Type[BaseModel]) -> str:
        tool_schema = cls.transform_simple_pydantic_schema_to_tool_schema(pydantic_model)

        return cls.tool_prompt_template.format(
            function_name=tool_schema["name"],
            function_description=tool_schema["description"],
            function_schema=json.dumps(tool_schema)
        )

    @classmethod
    def transform_simple_pydantic_schema_to_tool_schema(
        cls, pydantic_model: Type[BaseModel]
    ) -> dict:
        pydantic_schema = pydantic_model.model_json_schema()
        
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


class UserQueryParsing:
    def __init__(
        self, 
        asset_type: Literal["dataset", "ml_model"], 
        db_to_translate: Literal["Milvus"] = "Milvus",
        stage_1_fewshot_examples_filepath: str | None = None,
        stage_2_fewshot_examples_dirpath: str | None = None
    ) -> None:
        assert asset_type in ASSET_METADATA_SCHEMAS.keys(), f"Invalid asset_type argument '{asset_type}'"
        assert db_to_translate in ["Milvus"], f"Invalid db_to_translate argument '{db_to_translate}'"

        self.asset_type = asset_type
        self.db_to_translate = db_to_translate

        self.asset_schema = ASSET_METADATA_SCHEMAS[asset_type]
        self.stage_pipe_1 = UserQueryParsingStages.init_stage_1(
            asset_schema=self.asset_schema,
            fewshot_examples_path=stage_1_fewshot_examples_filepath
        )
        self.pipe_stage_2 = UserQueryParsingStages.init_stage_2(
            asset_schema=self.asset_schema,
            fewshot_examples_dirpath=stage_2_fewshot_examples_dirpath
        )

    def __call__(self, user_query: str) -> dict:
        out_stage_1 = self.stage_pipe_1({"query": user_query})
        if out_stage_1 is None:
            return {
                "query": user_query,
                "filter": ""
            }
        
        topic_list = [out_stage_1["topic"]]
        [
            topic_list.append(cond["condition"]) 
            for cond in out_stage_1["conditions"] 
            if cond.get("discard", False)
        ] # discarded conditions

        parsed_conditions = []
        valid_conditions = [cond for cond in out_stage_1["conditions"] if cond.get("discard", False) is False]
        for nl_cond in valid_conditions:
            if nl_cond["field"] not in self.asset_schema.model_fields.keys():
                topic_list.append(nl_cond["condition"])
                continue
            
            input = {    
                "condition": nl_cond["condition"],
                "field": nl_cond["field"]
            }
            out_stage_2 = self.pipe_stage_2(input)
            if out_stage_2 is None:
                topic = self.expand_query(topic, nl_cond["condition"])
                continue
        
            [
                topic_list.append(expr["raw_value"])
                for expr in out_stage_2["expressions"] 
                if expr.get("discard", False)
            ] # discarded expressions

            parsed_conditions.append({
                "field": nl_cond["field"],
                "logical_operator": out_stage_2["logical_operator"],
                "expressions": [
                    expr 
                    for expr in out_stage_2["expressions"] 
                    if expr.get("discard", False) is False
                ]
            })

        return (
            " ".join(topic_list),
            self.milvus_translate(parsed_conditions)
        )

    def milvus_translate(self, parsed_conditions: list[dict]) -> str:
        # TODO logic for translating conditions / expressions
        # There's an AND operation between all the items of the the 'parsed_conditions'
        # There may be AND/OR operation applied between expressions inside an item of the 'parsed_conditions'
            # There needs to be 2 or more expressions within one condition in order for it to work
        # Build filter string
            # It depends on whether its a list metadata field or a simple one

        # Some logic for translating is already found in the 'build_milvus_filter' function

        return parsed_conditions
        
        pass



class UserQueryParsingStages:
    task_instructions_stage1 = """
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

    task_instructions_stage2 = """
        Your task is to parse a single condition extracted from a user query and transform it into a structured format for further processing. The condition consists of one or more expressions combined with a logical operator.

        **Key Terminology:**
        1. **Expression**:
        - Represents a single comparison between a value and a metadata field.
        - Includes:
            - `raw_value`: The original value directly retrieved from the natural language condition.
            - `processed_value`: The transformed value, converted to the appropriate data type and format based on the value restrictions imposed on the metadata field '{field_name}'.
            - `comparison_operator`: The operator used for comparison (e.g., >, <, ==, !=).

        2. **Condition**:
        - Consists of one or more expressions combined with a logical operator.
        - Includes:
            - `expressions`: A list of expressions (at least one).
            - `logical_operator`: The logical relationship between expressions (AND/OR). This operator only makes sense when there are multiple expressions. If there's only one expression, set this to AND

        **Input:**
        On input You will receive:
        - `condition**: The natural language condition extracted from the user query. This query should contain one or more expressions to be extracted.

        **Instructions**:
        1. Identify potentionally all the expressions composing the condition. Each expression has its corresponding value and comparison_operator used to compare the value to metadata field for filtering purposes
        2. Make sure that you transform the value (processed_value) associated with each expression, so that it has the same data type and complies with the same restrictions as the ones applied to metadata field '{field_name}'. The metadata field description and the its value restrictions are the following: {field_description}
        3. Identify logical operator applied between expressions. There's only one operator (AND/OR) applied in between all expressions.
    """

    @classmethod
    def _create_dynamic_stage2_schema(cls, field_name: str, asset_schema: Type[BaseModel]) -> Type[BaseModel]:
        # TODO tied to datasets
        original_field = asset_schema.model_fields[field_name]
        
        expression_class = type(
            f"Expression_{field_name}",
            (BaseModel, ),
            {
                "__annotations__": {
                    "raw_value": str,
                    "processed_value": strip_list_type(strip_optional_type(original_field.annotation)),
                    "comparison_operator": Literal["<", ">", "<=", ">=", "==", "!="],
                },

                # We have intentionally split the value into two separate fields, into raw_value and processed value as our model had trouble
                # properly processing the values immediately. By defining an explicit intermediate step, to write down the raw value before transforming it,
                # we have actually managed to improve the model performance
                "raw_value": Field(..., description=f"The value used to compare to metadata field '{field_name}' in its raw state, extracted from the natural language condition"), 
                "processed_value": Field(..., description=f"The processed value used to compare to metadata field '{field_name}', that adheres to the same constraints as the field: {original_field.description}."),
                "comparison_operator": Field(..., description=f"The comparison operator that determines how the value should be compared to the metadata field '{field_name}'.")
            }
        )
        return type(
            f"Condition_{field_name}",
            (BaseModel, ),
            {
                "__annotations__": {
                    "expressions": list[expression_class],
                    "logical_operator": Literal["AND", "OR"]
                },
                "__doc__": f"Parsing of one condition pertaining to metadata field '{field_name}'. Condition comprises one or more expressions used to for filtering purposes",
                "expressions": Field(..., descriptions="List of expressions composing the entire condition. Each expression is associated with a particular value and a comparison operator used to compare the value to the metadata field."),
                "logical_operator": Field(..., descriptions="The logical operator that performs logical operations (AND/OR) between multiple expressions. If there's only one expression set this value to 'AND'.")
            }
        )

    @classmethod
    def _get_inner_most_primitive_type(cls, data_type: Type) -> Type:
        origin = get_origin(data_type)
        if origin is Literal:
            return type(get_args(data_type)[0])
        if origin is not None:
            args = get_args(data_type)
            if args: 
                return cls._get_inner_most_primitive_type(args[0])  # Check the first argument for simplicity
        return data_type

    @classmethod
    def _translate_primitive_type_to_str(cls, data_type: Type) -> str:
        if data_type not in [str, int, float]:
            raise ValueError("Not supported data type")
        return {
            str: "string",
            int: "integer",
            float: "float"
        }[data_type]
    
    @classmethod
    def _call_function_stage_1(
        cls, 
        chain: RunnableSequence, 
        input: dict,
    ) -> dict | None:
        return cls._try_invoke_stage_1(chain, input)
        
    @classmethod
    def _call_function_stage_2(
        cls, 
        chain: RunnableSequence, 
        input: dict, 
        asset_schema: Type[BaseModel],
        fewshot_examples_dirpath: str | None = None
    ) -> dict | None:
        metadata_field = input["field"]
        dynamic_type = cls._create_dynamic_stage2_schema(metadata_field, asset_schema)

        chain_to_use = chain
        if fewshot_examples_dirpath is not None:
            examples_path = os.path.join(fewshot_examples_dirpath, f"{metadata_field}.json")
            if os.path.exists(examples_path):
                with open(examples_path) as f:
                    fewshot_examples = json.load(f)
                if len(fewshot_examples) > 0:
                    example_prompt = ChatPromptTemplate.from_messages([
                        ("user", "Condition: {input}"),
                        ("ai", "{output}"),
                    ])
                    fewshot_prompt = FewShotChatMessagePromptTemplate(
                        examples=Llama_ManualFunctionCalling.transform_fewshot_examples(
                            dynamic_type, fewshot_examples
                        ),
                        example_prompt=example_prompt
                    )
                    old_prompt: ChatPromptTemplate = chain.steps[0]
                    new_prompt = ChatPromptTemplate.from_messages([
                        *old_prompt.messages[:-1],
                        fewshot_prompt,
                        old_prompt.messages[-1]
                    ])
                    chain_to_use = RunnableSequence(new_prompt, *chain.steps[1:])
                    

        input_variables = {
            "query": input["condition"],
            "field_name": metadata_field,
            "field_description": DatasetMetadataTemplate.model_fields[metadata_field].description,
            "system_prompt": Llama_ManualFunctionCalling.populate_tool_prompt(dynamic_type)
        }
        return cls._try_invoke_stage_2(chain_to_use, input_variables, dynamic_type)
    
    @classmethod
    def _try_invoke_stage_1(
        cls, chain: RunnableSequence, input: dict, num_retry_attempts: int = 5
    ) -> dict | None:
        best_output = None
        best_count = 0

        for _ in range(num_retry_attempts):
            output = chain.invoke(input)
            if output is None:
                continue
            try:
                SurfaceQueryParsing(**output)
                return output
            except:
                # TODO needs to be checked
                curr_count = 0
                for i in range(len(output["conditions"])):
                    try:
                        NaturalLanguageCondition(**output["conditions"][i])
                        curr_count += 1
                    except:
                        output["conditions"][i]["discard"] = True
                        continue
                if curr_count > best_count:
                    # check whether the entire object is correct once we get
                    # rid of invalid conditions

                    helper_object = deepcopy(output)
                    helper_object["conditions"] = [
                        cond 
                        for cond in output["conditions"]
                        if cond.get("discard", False) is False
                    ]
                    try:
                        SurfaceQueryParsing(**helper_object)
                        best_output = output
                        best_count = curr_count
                    except:
                        continue

        return best_output

    @classmethod
    def _try_invoke_stage_2(
        cls, chain: RunnableSequence, input: dict, wrapper_schema: Type[BaseModel], num_retry_attempts: int = 5
    ) -> dict | None:
        best_output = None
        best_count = 0

        expression_schema = strip_list_type(
            wrapper_schema.__annotations__["expressions"]
        )
        for _ in range(num_retry_attempts):
            output = chain.invoke(input)
            if output is None:
                continue
            try:
                wrapper_schema(**output)
                return output
            except:
                # TODO needs to be checked
                curr_count = 0
                for i in range(len(output["expressions"])):
                    try:
                        expression_schema(**output["expressions"][i])
                        curr_count += 1
                    except:
                        output["expressions"][i]["discard"] = True
                        continue
                if curr_count > best_count:
                    # check whether the entire object is correct once we get
                    # rid of invalid expressions

                    helper_object = deepcopy(output)
                    helper_object["expressions"] = [
                        expr 
                        for expr in output["expressions"]
                        if expr.get("discard", False) is False
                    ]
                    try:
                        wrapper_schema(**helper_object)
                        best_output = output
                        best_count = curr_count
                    except:
                        continue
            
        return best_output
        
    @classmethod
    def init_stage_1(
        cls, 
        asset_schema: Type[BaseModel],
        fewshot_examples_path: str | None = None, 
    ) -> Llama_ManualFunctionCalling:
        pydantic_model = SurfaceQueryParsing
        
        metadata_field_info = [
            {
                "name": name, 
                "description": field.description, 
                "type": cls._translate_primitive_type_to_str(cls._get_inner_most_primitive_type(field.annotation))
            } for name, field in asset_schema.model_fields.items()
        ]
        task_instructions = HumanMessagePromptTemplate.from_template(
            cls.task_instructions_stage1, 
            partial_variables={"model_schema": json.dumps(metadata_field_info)}
        )

        fewshot_prompt = ("user", "")
        if fewshot_examples_path is not None and os.path.exists(fewshot_examples_path):
            with open(fewshot_examples_path) as f:
                fewshot_examples = json.load(f)
            if len(fewshot_examples) > 0:
                example_prompt = ChatPromptTemplate.from_messages([
                    ("user", "User Query: {input}"),
                    ("ai", "{output}"),
                ])
                fewshot_prompt = FewShotChatMessagePromptTemplate(
                    examples=Llama_ManualFunctionCalling.transform_fewshot_examples(
                        pydantic_model, fewshot_examples
                    ),
                    example_prompt=example_prompt
                )
    
        chat_prompt_no_system = ChatPromptTemplate.from_messages([
            task_instructions,
            fewshot_prompt,
            ("user", "User Query: {query}"),
        ])
        return Llama_ManualFunctionCalling(
            model, 
            pydantic_model=pydantic_model, 
            chat_prompt_no_system=chat_prompt_no_system,
            call_function=cls._call_function_stage_1
        )
    
    @classmethod
    def init_stage_2(
        cls, 
        asset_schema: Type[BaseModel],
        fewshot_examples_dirpath: str | None = None
    ) -> Llama_ManualFunctionCalling:
        chat_prompt_no_system = ChatPromptTemplate.from_messages([
            ("user", cls.task_instructions_stage2),
            ("user", "Condition: {query}"),
        ])
        return Llama_ManualFunctionCalling(
            model, 
            pydantic_model=None, 
            chat_prompt_no_system=chat_prompt_no_system,
            call_function=partial(
                cls._call_function_stage_2, 
                asset_schema=asset_schema,
                fewshot_examples_dirpath=fewshot_examples_dirpath
            )
        )


if __name__ == "__main__":
    MODEL_NAME = "llama3.1:8b"
    model = ChatOllama(model=MODEL_NAME, num_predict=4096, num_ctx=8192)
    
    # user_query = (
    #     "Retrieve all the translation Stanford datasets that have more than 10k datapoints and fewer than 100k datapoints. The dataset has over 100k KB in size " +
    #     "and the dataset should contain one or more of the following languages: Slovak, Polish, French, German. However we don't wish the dataset to contain any English nor Spanish"
    # )
    user_query = (
        "Retrieve all the translation datasets that have more than 10k datapoints and fewer than 100k datapoints " +
        "and the dataset should contain one or more of the following languages: Chinese, Inidian."
    )
    # user_query = "I dont want news datasets with any textual or video data."
    # user_query = "I want datasets with Slovak or English. It also needs to have either French or German"


    # output_1stage = {
    #     "topic": 'translation Stanford datasets',
    #     "conditions": [{'condition': 'datasets with more than 10k datapoints and fewer than 100k datapoints', 'field': 'num_datapoints', 'operator': 'AND'}, {'condition': 'contain one or more of the following languages: Chinese, Inidian', 'field': 'languages', 'operator': 'OR'}]
    # }
    # input_2stage = [
    #     {    
    #         "condition": condition["condition"],
    #         "field": condition["field"],
    #     }
    #     for condition in output_1stage["conditions"]
    # ]

    # extraction_stage2 = UserQueryParsingStages.init_stage_2(asset_schema=DatasetMetadataTemplate)

    # outs = []
    # for inp in input_2stage:
    #     out = extraction_stage2(inp)
    #     outs.append(out)
        
 
    user_query_parsing = UserQueryParsing(
        asset_type="dataset",
        stage_1_fewshot_examples_filepath="src/fewshot_examples/user_query_stage1/stage1.json"
    )
    user_query_parsing(user_query)



    exit()

    #######################

    # from preprocess.text_operations import ConvertJsonToString
    # with open("temp/data_examples/huggingface.json") as f:
    #     data = json.load(f)[0]3
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


