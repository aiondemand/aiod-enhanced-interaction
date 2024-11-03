import json
import os
from typing import Generic, Literal, Union, Optional, Type, TypeVar, get_origin, get_args
from enum import Enum
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.callbacks import get_openai_callback
from langchain_core.output_parsers import (
    JsonOutputParser, StrOutputParser, BaseOutputParser
)
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.language_models.llms import BaseLLM
from torch.utils.data import DataLoader

from dataset import Queries
from lang_chains import ChainOutputOpts, SimpleChain, load_llm
from data_types import AnnotatedDoc, QueryDatapoint, SemanticSearchResult



class DatasetMetadataTemplate(BaseModel):
    # General metadata
    platform: Literal["huggingface", "openml", "zenodo"] = Field(
        ..., 
        description="The platform where the asset is hosted."
    )
    date_published: str = Field(
        ..., 
        description="The original publication date of the asset in the format 'YYYY-MM-DDTHH:MM:SS'."
    )
    year: int = Field(
        ..., 
        description="The year extracted from the publication date."
    )
    month: int = Field(
        ..., 
        description="The month extracted from the publication date."
    )
    domain: list[Literal["NLP", "Computer Vision", "Audio Processing"]] = Field(
        ..., 
        description="The AI technical domains of the asset, describing the type of data and AI task involved. Leave the list empty if not specified."
    )
    task_type: list[str] = Field(
        ..., 
        description="The machine learning tasks supported by this asset. Acceptable values include task types found on HuggingFace (e.g., 'token-classification', 'question-answering', ...). Leave the list empty if not specified"
    )
    license: Optional[str] = Field(
        None, 
        description="The license type governing the asset usage, if specified."
    )

    # Dataset-specific metadata
    size_in_mb: Optional[float] = Field(
        None, 
        gt=0,
        description="The total size of the dataset in megabytes. If the size is not explicitly specified in the dataset descritpion, sum up the sizes of individual files instead if possible. Don't forget to convert the sizes to MBs"
    )
    num_datapoints: Optional[float] = Field(
        None, 
        gt=0,
        description="The number of data points in the dataset, if specified."
    )
    size_category: Optional[str] = Field(
        None, 
        description="The general size category of the dataset, typically specified in ranges such as '10k<n<100k' found on HuggingFace. If you know the precise number of datapoints you may infer the size category."
    )
    modality: list[Literal["text", "tabular", "audio", "video", "image"]] = Field(
        ..., 
        description="The modalities present in the dataset, such as 'text', 'tabular', 'audio', 'video', or 'image'. Leave the list empty if not specified"
    )
    data_format: list[str] = Field(
        ..., 
        description="The file formats of the dataset (e.g., 'CSV', 'JSON', 'Parquet'), if specified. Leave the list empty if not specified"
    )
    languages: list[str] = Field(
        ..., 
        description="Languages present in the dataset, specified in ISO 639-1 two-letter codes (e.g., 'EN' for English, 'ES' for Spanish, 'FR' for French, ...). The description of the dataset itself has no relation to the languages present in the dataset we wish to retrieve. Leave the list empty if not specified"
    )


def is_optional_type(annotation: Type) -> bool:
    if get_origin(annotation) is Union:
        return type(None) in get_args(annotation)
    return False


def metadata_wrapper_type_factory(
    template_type: Type[BaseModel], 
    for_user_query: bool = False
) -> Type[BaseModel]:
    type_name = f'MetadataType_{"FromUserQuery" if for_user_query else "FromAssets"}'
    annotations = template_type.__annotations__
    if for_user_query:
        # make all the types/fields optional
        annotations = {
            k: v if is_optional_type(v) else Optional[v]
            for k, v in annotations.items()
        }

    new_fields = {
        k: 
            Field(None, description=v.field_info.description) 
            if is_optional_type(annotations[k]) 
            else Field(..., description=v.field_info.description)
        for k, v in template_type.__fields__.items()
    }
    
    # wrap each attribute to Field/Condition wrapper
    annotations = {
        k: metadata_field_type_factory(
            parent_field_name=k, 
            parent_description=new_fields[k].description,
            data_type=v, 
            for_user_query=for_user_query
        )
        for k, v in annotations.items()
    }

    arguments = {
        '__annotations__': annotations,
    }
    arguments.update(new_fields)
    return type(type_name, (BaseModel, ), arguments)


def metadata_field_type_factory(
    parent_field_name: str, 
    parent_description: str,
    data_type: Type[BaseModel], 
    for_user_query: bool = False
) -> Type[BaseModel]:
    type_name = f'MetadataField_{parent_field_name}_{"FromUserQuery" if for_user_query else "FromAssets"}'
    if for_user_query:
        return type(
            type_name,
            (BaseModel, ),
            {
                '__annotations__': {
                    "value": data_type,
                    "operator": Literal["<", ">", "<=", ">=", "==", "!=", "IN", "NOT IN"]
                },
                'value': Field(..., description=f"Value found in the user query corresponding to metadata field:"),
                'operator': Field(..., description="Comparison operator used to create a condition/filter together with the extracted metadata value found in the user query. If the metadata field can have multiple values, use operators `IN` or `NOT IN` exclusively."),
            }
        )
    return type(
        type_name,
        (BaseModel, ),
        {
            '__annotations__': {
                "value": data_type,
            },
            'value': Field(..., description=f"Value of the metadata field: {parent_description}"),
        }
    )


DatasetMetadata_FromAssets_Schema: Type[BaseModel] = metadata_wrapper_type_factory(
    template_type=DatasetMetadataTemplate, 
    for_user_query=False
)


DatasetMetadata_FromUserQueries_Schema: Type[BaseModel] = metadata_wrapper_type_factory(
    template_type=DatasetMetadataTemplate, 
    for_user_query=True
)



class LLM_MetadataExtractor:
    system_prompt_from_asset = """
        ### Task Overview:
        You are tasked with extracting specific metadata attributes from a document describing a machine learning {asset_type}. This metadata will be used for database filtering, so precision, clarity, and high confidence are essential. The extraction should focus only on values you are highly confident in, ensuring they are concise, unambiguous, and directly relevant.

        ### Instructions:
        - Extract Only High-Confidence Metadata: Only extract values if you are certain of their accuracy based on the provided document. If you are unsure of a particular metadata value, skip the attribute and set it to None.
        - Align Values with Controlled Terminology: Some extracted values may need minor modifications to match acceptable or standardized terminology (e.g., specific task types or domains). Ensure that values align with controlled terminology, making them consistent and unambiguous.
        - Structured and Minimal Output: The extracted metadata must align strictly with the provided schema. Avoid adding extra information or interpretations that are not directly stated in the document. If multiple values are relevant for a field, list them as specified by the schema.
        - Focus on Database Usability: The extracted metadata will be used to categorize and filter {asset_type}s in a database, so each value should be directly useful for this purpose. This means avoiding ambiguous or verbose values that would complicate search and filtering.
    """
    system_prompt_from_user_query = """
        ### Task Overview:
        You are an advanced language model skilled in extracting structured filter conditions from natural language user queries. 
        Your task is to identify filter conditions for predefined set of metadata fields that are a part of JSON schema defined at the end of this prompt 
        representing the form your output needs to adhere to.

        Each condition must include the following components:
        1. **Operator**: The comparison operator that determines how the value should be evaluated (e.g., `=`, `>`, `<`, `!=`, `IN`, `NOT IN`).
        2. **Value**: The specific value associated with the filter condition.

        These components must be mapped to a valid metadata field found in the JSON schema below. 
        If a condition cannot be fully constructed (i.e., is missing either the operator or value), do not include it in the output.

        ### Examples:

        **User Query 1**: "Find all chocolate datasets created after January 1, 2022, that are marked as high-priority and with a size smaller than 500 000KB."

        **Output**:
        - Condition tied to `date_published` metadata field:
            - Operator: >
            - Value: 2022-01-01T00:00:00
            - Note: Notice that we needed to convert the extracted value to corresponding datatype and format of the metadata
        - Condition tied to `size_in_mb` metadata field:
            - Operator: <
            - Value: 488
            - Note: We have once again converted the value, this time from KB to MB to align the units properly
        - Note: Since we don't track `priority` as a metadata field, we don't create a condition


        **User Query 2**: "Show me the French cross words datasets created in the past year."

        **Output**:
        - Condition tied to `languages` metadata field:
            - Operator: IN
            - Value: FR
            - Note: We have selected the `IN` operator, since the `languages` metadata field has a form of a list of multiple possible values. 
            The `IN` condition checks whether the requested value is found in the list of values of the dataset. 
            The word French was also converted to accomodate the formatting of the said metadata field.
        
        - Note: We don't know the specific date "the past year" pertains to, hence the omission of this condition, even though we do in fact track the `date_published` metadata

        **Output**:
        - Metadata Field: project, Operator: =, Value: "Alpha"
        - Metadata Field: name, Operator: contains, Value: "report"
    """

    user_prompt_from_asset = """
        ### Document of machine learning {asset_type} to extract metadata from:
        {document}
    
        ### Output format:
        {format}
    """
    user_prompt_from_user_query = """
        ### User query containing potential conditions to extract and filter machine learning {asset_type}s on:
        {query}
    
        ### Output format:
        {format}
    """

    @classmethod
    def build_chain(
        cls, 
        llm: BaseLLM | None = None,
        pydantic_model: Type[BaseModel] | None = None,
        asset_type: Literal["dataset", "model"] = "dataset",
        extracting_from_user_query: bool = False
    ) -> SimpleChain:
        if llm is None:
            llm = load_llm()
        if pydantic_model is None:
            pydantic_model = (
                DatasetMetadata_FromUserQueries_Schema
                if extracting_from_user_query
                else DatasetMetadata_FromAssets_Schema
            )
        system_prompt = (
            cls.system_prompt_from_user_query 
            if extracting_from_user_query 
            else cls.system_prompt_from_asset
        )
        user_prompt = (
            cls.user_prompt_from_user_query 
            if extracting_from_user_query 
            else cls.user_prompt_from_asset
        )
        prompt_templates = [
            system_prompt.format(asset_type=asset_type), 
            user_prompt
        ]

        use_openai_bind_tools = isinstance(llm, BaseChatOpenAI)
        postprocess_lambda = lambda out: out[0] if use_openai_bind_tools else None

        chain_output_opts = ChainOutputOpts(
            langchain_parser_class=JsonOutputParser,
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
    
    def __init__(
        self, chain: SimpleChain | None = None, 
        asset_type: Literal["dataset", "model"] = "dataset",
        extracting_from_user_query: bool = False
    ) -> None:
        self.asset_type = asset_type
        self.extracting_from_user_query = extracting_from_user_query
        
        self.chain = chain
        if chain is None:
            self.chain = self.build_chain(
                asset_type=asset_type,
                extracting_from_user_query=extracting_from_user_query
            )

    def __call__(self, input: dict) -> dict | str | None:
        input = input.copy()
        input["asset_type"] = self.asset_type
        return self.chain.invoke(input)


if __name__ == "__main__":    
    # from preprocess.text_operations import ConvertJsonToString
    # with open("temp/data_examples/zenodo.json") as f:
    #     data = json.load(f)[0]
    # text_format = ConvertJsonToString().extract_relevant_info(data)
    
    # extract = LLM_MetadataExtractor()
    # output = extract({"document": text_format})
    # output

    user_query = "Retrieve all the summarization datasets with at least 10k datapoints and has either Slovak or French language in them."
    extract_query = LLM_MetadataExtractor(asset_type="dataset", extracting_from_user_query=True)

    output = extract_query({"query": user_query})
    output
    




