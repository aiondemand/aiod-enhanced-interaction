from copy import deepcopy
from functools import partial
from ast import literal_eval
import os
import json
import re
from typing import Any, Callable, ClassVar, Literal, Union, Optional, Type, get_origin, get_args
from enum import Enum
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, field_validator
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import FewShotChatMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.language_models.llms import BaseLLM
from langchain_core.runnables import RunnableLambda, RunnableSequence

from lang_chains import LLM_Chain, SimpleChain, load_llm


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
            

class HuggingFaceDatasetMetadataTemplate(BaseModel):
    """
    Extraction of relevant metadata we wish to retrieve from ML assets
    """

    _ALL_VALID_VALUES: ClassVar[list[list[str]] | None] = None

    date_published: str = Field(
        ..., 
        description="The publication date of the dataset in the format 'YYYY-MM-DDTHH:MM:SSZ'."
    )
    size_in_mb: Optional[int] = Field(
        None, 
        description="The total size of the dataset in megabytes. Don't forget to convert the sizes to MBs if necessary.",
        ge=0,
    )
    license: Optional[str] = Field(
        None, 
        description="The license associated with this dataset, e.g., 'mit', 'apache-2.0'"
    )
    task_types: Optional[list[str]] = Field(
        None, 
        description="The machine learning tasks suitable for this dataset. Acceptable values may include task categories or task ids found on HuggingFace platform (e.g., 'token-classification', 'question-answering', ...)"
    )
    languages: Optional[list[str]] = Field(
        None, 
        description="Languages present in the dataset, specified in ISO 639-1 two-letter codes (e.g., 'en' for English, 'es' for Spanish, 'fr' for French, etc ...)."
    )
    datapoints_lower_bound: Optional[int] = Field(
        None,
        description="The lower bound of the number of datapoints in the dataset. This value represents the minimum number of datapoints found in the dataset."
    )
    datapoints_upper_bound: Optional[int] = Field(
        None,
        description="The upper bound of the number of datapoints in the dataset. This value represents the maximum number of datapoints found in the dataset."
    )

    @classmethod
    def _load_all_valid_values(cls) -> None:
        # TODO get rid of ugly path
        path = "src/preprocess/hf_dataset_metadata_values.json"
        with open(path) as f:
            cls._ALL_VALID_VALUES = json.load(f)
    
    @classmethod
    def get_field_valid_values(cls, field: str) -> list[str]:
        if cls._ALL_VALID_VALUES is None:
            cls._load_all_valid_values()
        return cls._ALL_VALID_VALUES.get(field, None)
        
    @classmethod
    def exists_field_valid_values(cls, field: str) -> bool:
        if cls._ALL_VALID_VALUES is None:
            cls._load_all_valid_values()
        return field in cls._ALL_VALID_VALUES.keys()

    @classmethod
    def validate_value_against_list(cls, val: str, field: str) -> bool:
        if cls.exists_field_valid_values(field) is False:
            return True
        return val in cls.get_field_valid_values(field)

    @classmethod
    def _check_field_against_list_wrapper(cls, values: list[str], field: str) -> list[str]:
        valid_values = [
            val.lower() for val in values
            if cls.validate_value_against_list(val, field)
        ]
        if len(valid_values) == 0:
            return None
        return valid_values
    
    @field_validator("date_published", mode="before")
    @classmethod
    def check_date_published(cls, value: str) -> str | None:
        pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
        return bool(re.match(pattern, value))
    
    @field_validator("license", mode="before")
    @classmethod
    def check_license(cls, values: list[str]) -> list[str] | None:
        return cls._check_field_against_list_wrapper(values, "license")
    
    @field_validator("task_types", mode="before")
    @classmethod
    def check_task_types(cls, values: list[str]) -> list[str] | None:
        return cls._check_field_against_list_wrapper(values, "task_types")
    
    @field_validator("languages", mode="before")
    @classmethod
    def check_languages(cls, values: list[str]) -> list[str] | None:
        return [val.lower() for val in values if len(val) == 2]
                

# Classes representing specific asset metadata 
class OldDatasetMetadataTemplate(BaseModel):
    """
    Extraction of relevant metadata we wish to retrieve from ML assets
    """
    
    platform: Literal["huggingface", "openml", "zenodo"] = Field(
        ..., 
        description="The platform where the asset is hosted. Only permitted values: ['huggingface', 'openml', 'zenodo']"
    )
    date_published: str = Field(
        ..., 
        description="The original publication date of the asset in the format 'YYYY-MM-DDTHH-MM-SSZ'."
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

    def __call__(
        self, input: dict | list[dict], as_batch: bool = False
    ) -> dict | None | list[dict | None]:
        if self.call_function is not None:
            return self.call_function(self.chain, input)

        if as_batch is False:
            out = self.chain.invoke(input)
            if self.validate_output(out, self.pydantic_model):
                return out
            return None
        else:
            out = self.chain.batch(input)
            validated_out = [
                o if self.validate_output(o, self.pydantic_model) else None
                for o in out
            ]
            return validated_out

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
            _, args_string = match.groups()
            try:
                return literal_eval(args_string)
            except:
                try:
                    return json.loads(args_string)
                except:
                    pass
        
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


# Old implementation of asset metadata extraction
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
                     template_type=OldDatasetMetadataTemplate
                ) 
                if parsing_user_query
                else OldDatasetMetadataTemplate
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
    SCHEMA_MAPPING = {
        "datasets": HuggingFaceDatasetMetadataTemplate
    }
    
    @classmethod
    def get_asset_schema(cls, asset_type: str) -> Type[BaseModel]:
        return cls.SCHEMA_MAPPING[asset_type]
    
    def __init__(
        self, 
        llm: BaseLLM,
        asset_type: Literal["datasets"],
        db_to_translate: Literal["Milvus"] = "Milvus",
        stage_1_fewshot_examples_filepath: str | None = None,
        stage_2_fewshot_examples_dirpath: str | None = None
    ) -> None:
        assert asset_type in self.SCHEMA_MAPPING.keys(), f"Invalid asset_type argument '{asset_type}'"
        assert db_to_translate in ["Milvus"], f"Invalid db_to_translate argument '{db_to_translate}'"

        self.asset_type = asset_type
        self.db_to_translate = db_to_translate

        self.asset_schema = self.get_asset_schema(asset_type)
        self.stage_pipe_1 = UserQueryParsingStages.init_stage_1(
            llm=llm,
            asset_schema=self.asset_schema,
            fewshot_examples_path=stage_1_fewshot_examples_filepath
        )
        self.pipe_stage_2 = UserQueryParsingStages.init_stage_2(
            llm=llm,
            asset_schema=self.asset_schema,
            fewshot_examples_dirpath=stage_2_fewshot_examples_dirpath
        )

    def __call__(self, user_query: str) -> dict:        
        def expand_topic_query(
            topic_list: list[str], new_objects: list[dict], content_key: str
        ) -> list[str]:
            to_add = [
                item[content_key]
                for item in new_objects
                if (
                    item.get("discard", "false") == "true" and 
                    item.get(content_key, None) is not None
                )
            ]
            return topic_list + to_add
        
        
        out_stage_1 = self.stage_pipe_1({"query": user_query})
        if out_stage_1 is None:
            return {
                "query": user_query,
                "filter": ""
            }
        
        topic_list = [out_stage_1["topic"]]
        topic_list = expand_topic_query(
            topic_list, out_stage_1["conditions"], content_key="condition"
        )

        parsed_conditions = []
        valid_conditions = [
            cond for cond in out_stage_1["conditions"] 
            if cond.get("discard", "false") == "false"
        ]
        for nl_cond in valid_conditions:    
            input = {    
                "condition": nl_cond["condition"],
                "field": nl_cond["field"]
            }
            out_stage_2 = self.pipe_stage_2(input)
            if out_stage_2 is None:
                topic_list.append(nl_cond["condition"])
                continue
        
            topic_list = expand_topic_query(
                topic_list, out_stage_2["expressions"], content_key="raw_value"
            )

            valid_expressions = [
                expr for expr in out_stage_2["expressions"] 
                if expr.get("discard", "false") == "false"
            ]
            if len(valid_expressions) > 0:
                parsed_conditions.append({
                    "field": nl_cond["field"],
                    "logical_operator": out_stage_2["logical_operator"],
                    "expressions": valid_expressions
                })

        topic = " ".join(topic_list)
        filter_string = self.milvus_translate(parsed_conditions)

        return topic, filter_string

    def milvus_translate(self, parsed_conditions: list[dict]) -> str:
        simple_expression_template = "({field} {op} {val})"
        list_expression_template = "({op}ARRAY_CONTAINS({field}, {val}))"
        format_value = lambda x: f"'{x.lower()}'" if isinstance(x, str) else x
        list_fields = {
            k: get_origin(strip_optional_type(v)) is list
            for k, v in self.asset_schema.__annotations__.items()
        }

        condition_strings = []
        for parsed_condition in parsed_conditions:
            field = parsed_condition["field"]
            log_operator = parsed_condition["logical_operator"]
            
            str_expressions = []
            for expr in parsed_condition["expressions"]:
                comp_operator = expr["comparison_operator"]
                val = expr["processed_value"]

                if list_fields[field]:
                    if comp_operator not in ["==", "!="]:
                        raise ValueError(
                            "We don't support any other comparison operators but a '==', '!=' for checking whether values exist whithin the metadata field."
                        )
                    str_expressions.append(
                        list_expression_template.format(
                            field=field,
                            op="" if comp_operator == "==" else "not ",
                            val=format_value(val)
                        )
                    )
                else:
                    str_expressions.append(
                        simple_expression_template.format(
                            field=field,
                            op=comp_operator,
                            val=format_value(val)
                        )
                    )
            condition_strings.append(
                "(" + f" {log_operator.lower()} ".join(str_expressions) + ")"
            )

        return " and ".join(condition_strings)


class UserQueryParsingStages:
    task_instructions_stage1 = """
        Your task is to process a user query that may contain multiple natural language conditions and a general topic. Each condition may correspond to a specific metadata field and describes one or more values that should be compared against that field.
        These conditions are subsequently used to filter out unsatisfactory data in database. On the other hand, the topic is used in semantic search to find the most relevant documents to the thing user seeks for
        
        A simple schema below briefly describes all the metadata fields we use for filtering purposes:
        {model_schema}

        **Key Guidelines:**

        1. **Conditions and Metadata Fields:**
        - Each condition must clearly correspond to exactly one metadata field that we use for filtering purposes.
        - If a condition is associated with a metadata field not found in the defined schema, disregard that condition.
        - If a condition references multiple metadata fields (e.g., "published after 2020 or in image format"), split it into separate conditions with NONE operators, each tied to its respective metadata field.

        2. **Handling Multiple Values:**
        - If a condition references multiple values for a single metadata field (e.g., "dataset containing French or English"), include all the values in the natural language condition.
        - Specify the logical operator (AND/OR) that ties the values:
            - Use **AND** when the query requires all the values to match simultaneously.
            - Use **OR** when the query allows any of the values to match.

        3. **Natural Language Representation:**
        - Preserve the natural language form of the conditions. You're also allowed to modify them slightly to preserve their meaning once they're extracted from their context

        4. **Logical Operators for Conditions:**
        - Always include a logical operator (AND/OR) for conditions with multiple values.
        - For conditions with a single value, the logical operator is not required but can default to "NONE" for clarity.

        5. **Extract user query topic**
        - A topic is a concise, high-level description of the main subject of the query. 
        - It should exclude specific filtering conditions but still capture the core concept or intent of the query.
        - If the query does not explicitly state a topic, infer it based on the overall context of the query.
    """

    task_instructions_stage2 = """
        Your task is to parse a single condition extracted from a user query and transform it into a structured format for further processing. The condition consists of one or more expressions combined with a logical operator.        
        Validate whether each expression value can be unambiguously transformed into its processed valid counterpart compliant with the restrictions imposed on the metadata field. If transformation of the expression value is not clear and ambiguous, discard the expression instead.

        **Key Terminology:**
        1. **Expression**:
        - Represents a single comparison between a value and a metadata field.
        - Includes:
            - `raw_value`: The original value directly retrieved from the natural language condition.
            - `processed_value`: The transformed `raw_value`, converted to the appropriate data type and format based on the value and type restrictions imposed on the metadata field '{field_name}'. If `raw_value` cannot be unambiguously converted to a its valid counterpart complaint with metadata field constraints, set this field to string value "NONE".
            - `comparison_operator`: The operator used for comparison (e.g., >, <, ==, !=).
            - `discard`: A boolean value indicating whether the expression should be discarded (True if `raw_value` cannot be unambiguously transformed into a valid `processed_value`).

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
        2. Make sure that you perform an unambiguous transformation of the raw value associated with each expression to its valid counterpart that is compliant with the restrictions imposed on the metadata field '{field_name}'. The metadata field description and the its value restrictions are the following: 
            a) Description: {field_description}
            {field_valid_values}
            
            If the transformation of the raw value is not clear and ambiguous, discard the expression.
        3. Identify logical operator applied between expressions. There's only one operator (AND/OR) applied in between all expressions.
    """

    @classmethod
    def _create_dynamic_stage2_schema(cls, field_name: str, asset_schema: Type[BaseModel]) -> Type[BaseModel]:
        def validate_func(cls, value: Any, func: Callable) -> Any:
            if value == "NONE" or func(value):
                return value        
            raise ValueError(
                f"Invalid processed value"
            )
        
        original_field = asset_schema.model_fields[field_name]    
        inner_class_dict = {
            "__annotations__": {
                "raw_value": str,
                "processed_value": strip_list_type(strip_optional_type(original_field.annotation)) | Literal["NONE"],
                "comparison_operator": Literal["<", ">", "<=", ">=", "==", "!="],
                "discard": Literal["false", "true"]
            },

            # We have intentionally split the value into two separate fields, into raw_value and processed value as our model had trouble
            # properly processing the values immediately. By defining an explicit intermediate step, to write down the raw value before transforming it,
            # we have actually managed to improve the model performance
            "raw_value": Field(..., description=f"The value used to compare to metadata field '{field_name}' in its raw state, extracted from the natural language condition"), 
            "processed_value": Field(..., description=f"The processed value used to compare to metadata field '{field_name}', that adheres to the same constraints as the field: {original_field.description}."),
            "comparison_operator": Field(..., description=f"The comparison operator that determines how the value should be compared to the metadata field '{field_name}'."),
            "discard": Field("false", description="A boolean value indicating whether the expression should be discarded if 'raw_value' cannot be transformed into a valid 'processed_value'"),
        }
        
        validators = [
            (func_name, decor)
            for func_name, decor in asset_schema.__pydantic_decorators__.field_validators.items()
            if field_name in decor.info.fields
        ]
        if len(validators) > 0:
            # we will accept only one decorator/validator for a field
            validator_func_name, decor = validators[0]

            inner_class_dict.update({
                # Validator for 'processed_value' attribute against all valid values
                "validate_processed_value": field_validator(
                    "processed_value", 
                    mode=decor.info.mode
                )(
                    partial(
                        validate_func, 
                        func=getattr(asset_schema, validator_func_name)
                    )
                )
            })
            
        expression_class = type(
            f"Expression_{field_name}",
            (BaseModel, ),
            inner_class_dict
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
        asset_schema: Type[BaseModel]
    ) -> dict | None:
        return cls._try_invoke_stage_1(chain, input, asset_schema)
        
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
                    
        field_valid_values = (
            f"b) List of the only permitted values: {asset_schema.get_field_valid_values(metadata_field)}"
            if asset_schema.exists_field_valid_values(metadata_field)
            else ""
        )
        input_variables = {
            "query": input["condition"],
            "field_name": metadata_field,
            "field_description": asset_schema.model_fields[metadata_field].description,
            "field_valid_values": field_valid_values,
            "system_prompt": Llama_ManualFunctionCalling.populate_tool_prompt(dynamic_type)
        }
        return cls._try_invoke_stage_2(chain_to_use, input_variables, dynamic_type)
    
    @classmethod
    def _try_invoke_stage_1(
        cls, chain: RunnableSequence, input: dict, 
        asset_schema: Type[BaseModel], 
        num_retry_attempts: int = 5
    ) -> dict | None:
        def exists_conditions_list_in_wrapper_dict(obj: dict) -> bool:
            return (
                obj.get("conditions", None) is not None and 
                isinstance(obj["conditions"], list)
            )
            
        def is_valid_wrapper_class(obj: dict, valid_field_names: list[str]) -> bool:
            try:
                SurfaceQueryParsing(**obj)

                invalid_fields = [
                    cond["field"] for cond in obj["conditions"]
                    if cond["field"] not in valid_field_names
                ]
                if len(invalid_fields) == 0:
                    return True
            except:
                pass
            return False

        def is_valid_condition_class(obj: Any, valid_field_names: list[str]) -> bool:            
            if isinstance(obj, dict) is False:
                return False
            try:
                NaturalLanguageCondition(**obj)

                if obj["field"] in valid_field_names:
                    return True
            except:
                pass
            return False
            
        best_llm_response = None
        max_valid_conditions_count = 0

        for _ in range(num_retry_attempts):
            output = chain.invoke(input)
            if output is None:
                continue
            valid_field_names = list(asset_schema.model_fields.keys())
            if is_valid_wrapper_class(output, valid_field_names):
                return SurfaceQueryParsing(**output).model_dump()
        
            # The LLM output is invalid, now we will identify 
            # which conditions are incorrect and how many are valid
            if exists_conditions_list_in_wrapper_dict(output) == False:
                continue
            valid_conditions_count = 0
            for i in range(len(output["conditions"])):
                if is_valid_condition_class(output["conditions"][i], valid_field_names):
                    valid_conditions_count += 1
                elif isinstance(output["conditions"][i], dict):
                    output["conditions"][i]["discard"] = "true"
                else:
                    output["conditions"][i] = { "discard": "true" }
                
            # we compare current LLM output to potentionally previous LLm outputs
            # and identify the best LLM response (containing the most valid conditions) 
            if valid_conditions_count > max_valid_conditions_count:
                # check whether the entire object is correct once we get
                # rid of invalid conditions
                helper_object = deepcopy(output)
                helper_object["conditions"] = [
                    cond for cond in output["conditions"]
                    if cond.get("discard", "false") == "false"
                ]
                if is_valid_wrapper_class(helper_object, valid_field_names):
                    best_llm_response = SurfaceQueryParsing(**output).model_dump()
                    max_valid_conditions_count = valid_conditions_count

        return best_llm_response

    @classmethod
    def _try_invoke_stage_2(
        cls, chain: RunnableSequence, input: dict, wrapper_schema: Type[BaseModel], num_retry_attempts: int = 5
    ) -> dict | None:
        def exists_expressions_list_in_wrapper_dict(obj: dict) -> bool:
            return (
                obj.get("expressions", None) is not None and 
                isinstance(obj["expressions"], list)
            )

        def is_valid_wrapper_class(obj: dict) -> bool:
            try:
                wrapper_schema(**obj)
                return True
            except:
                return False
        
        def is_valid_expression_class(obj: dict) -> bool:
            try:
                expression_schema(**obj)
                return True
            except:
                return False
        
        best_llm_response = None
        max_valid_expressions_count = 0
        expression_schema = strip_list_type(
            wrapper_schema.__annotations__["expressions"]
        )

        for _ in range(num_retry_attempts):
            output = chain.invoke(input)
            if output is None:
                continue
            if is_valid_wrapper_class(output):
                return wrapper_schema(**output).model_dump()
            
            # The LLM output is invalid, now we will identify 
            # which expressions are incorrect and how many are valid
            if exists_expressions_list_in_wrapper_dict(output) == False:
                continue
            valid_expressions_count = 0
            for i in range(len(output["expressions"])):
                if is_valid_expression_class(output["expressions"][i]):
                    valid_expressions_count += 1
                elif isinstance(output["expressions"][i], dict):
                    output["expressions"][i]["discard"] = "true"
                else:
                    output["expressions"][i] = { "discard": "true" }
                
            # we compare current LLM output to potentionally previous LLm outputs
            # and identify the best LLM response (containing the most valid expressions) 
            if valid_expressions_count > max_valid_expressions_count:
                # check whether the entire object is correct once we get
                # rid of invalid expressions
                helper_object = deepcopy(output)
                helper_object["expressions"] = [
                    expr for expr in output["expressions"]
                    if expr.get("discard", "false") == "false"
                ]
                if is_valid_wrapper_class(helper_object):
                    best_llm_response = wrapper_schema(**output).model_dump()
                    max_valid_expressions_count = valid_expressions_count
            
        return best_llm_response
        
    @classmethod
    def init_stage_1(
        cls, 
        llm: BaseLLM,
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
            llm, 
            pydantic_model=pydantic_model, 
            chat_prompt_no_system=chat_prompt_no_system,
            call_function=partial(
                cls._call_function_stage_1, 
                asset_schema=asset_schema
            )
        )
    
    @classmethod
    def init_stage_2(
        cls, 
        llm: BaseLLM,
        asset_schema: Type[BaseModel],
        fewshot_examples_dirpath: str | None = None
    ) -> Llama_ManualFunctionCalling:
        chat_prompt_no_system = ChatPromptTemplate.from_messages([
            ("user", cls.task_instructions_stage2),
            ("user", "Condition: {query}"),
        ])
        return Llama_ManualFunctionCalling(
            llm, 
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
    model = ChatOllama(model=MODEL_NAME, num_predict=1_024, num_ctx=4_096)
    
    # user_query = (
    #     "Retrieve all the translation datasets that have more than 20k datapoints and fewer than 80k datapoints " +
    #     "and the dataset should contain one or more of the following languages: Chinese, Iniiidian, Japanese. The data is stored in CSV files"
    # )
    # user_query = (
    #     "Retrieve all the summarization datasets from years 2022 and 2023 in CSV format that have more than 20k datapoints, but fewer than 480 000."
    # )

    user_query = (
        "Retrieve all the claim-matching datasets that are older than March of 2022 and that have more than 20k datapoints, but fewer than 5 million."
    )

    # user_query = (
    #     "Retrieve all datasets that tackle the problem of machine translation, with English, French and German data"
    # )

    user_query_parsing = UserQueryParsing(
        model,
        asset_type="datasets",
        stage_1_fewshot_examples_filepath="src/fewshot_examples/user_query_stage1/stage1.json"
    )
    topic, str_filter = user_query_parsing(user_query)

    exit()