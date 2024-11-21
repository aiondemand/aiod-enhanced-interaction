from ast import literal_eval
from tqdm import tqdm
import json
from typing import Type
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from llm_metadata_filter import DatasetMetadataTemplate, LLM_MetadataExtractor, user_query_metadata_extraction_schema_factory
from preprocess.text_operations import ConvertJsonToString

import re
import json





    

def transform_simple_pydantic_schema_to_tool_schema(pydantic_model: Type[BaseModel]) -> dict:
    pydantic_schema = pydantic_model.model_json_schema()
    
    pydantic_schema.pop("type")
    pydantic_schema["name"] = pydantic_schema.pop("title")
    pydantic_schema["parameters"] = {
        "type": "object",
        "properties": pydantic_schema.pop("properties"),
        "required": pydantic_schema.pop("required")
    }

    return pydantic_schema


def transform_nested_pydantic_schema_to_tool_schema(pydantic_model: Type[BaseModel]) -> dict:
    pydantic_schema = pydantic_model.model_json_schema()
    
    pydantic_schema.pop("type")
    pydantic_schema["name"] = pydantic_schema.pop("title")
    pydantic_schema["parameters"] = {
        "type": "object",
        "properties": pydantic_schema.pop("properties"),
        "required": []
    }

    return pydantic_schema


def test_asset_extraction():
    with open("temp/data_examples/huggingface.json") as f:
        data = json.load(f)[0]
    text_format = ConvertJsonToString().extract_relevant_info(data)

    pydantic_model = DatasetMetadataTemplate
    tool_schema = transform_simple_pydantic_schema_to_tool_schema(pydantic_model)
    tool_prompt = populate_tool_prompt(tool_schema)
    
    system_content = LLM_MetadataExtractor.system_prompt_from_asset.format(asset_type="dataset")
    user_content = LLM_MetadataExtractor.user_prompt_from_asset.format(
        asset_type="dataset",
        document=text_format
    )
    messages = [
        {
            "role": "system",
            "content": tool_prompt
        },
        {
            "role": "user",
            "content": system_content + "\n\n" + user_content
        }
    ]
    import ollama
    response = ollama.chat(
        model=MODEL_NAME,
        options={
            "num_ctx": 16_384,
        },
        messages=messages
    )
    
    tool_call = convert_llm_string_output_to_tool(response["message"]["content"])
    try:
        pydantic_model(**tool_call["arguments"])
    except:
        return None
    return tool_call["arguments"]


def test_query_extraction():
    user_query = (
        "Retrieve all the summarization datasets with at least 10k datapoints, yet no more than 100k datapoints, " +
        "and the dataset should have contain Slovak language or Polish language, but no Czech language."
    )

    # second stage

    pydantic_model = user_query_metadata_extraction_schema_factory(
        template_type=DatasetMetadataTemplate
    ) 
    # pydantic_model = UserQueryConditions
    
    tool_schema = transform_nested_pydantic_schema_to_tool_schema(pydantic_model)
    tool_prompt = populate_tool_prompt(tool_schema)

    system_content = LLM_MetadataExtractor.system_prompt_from_user_query.format(asset_type="dataset")
    user_content = LLM_MetadataExtractor.user_prompt_from_user_query.format(
        asset_type="dataset",
        query=user_query
    )
    messages = [
        {
            "role": "system",
            "content": tool_prompt
        },
        {
            "role": "user",
            "content": system_content + "\n\n" + user_content
        }
    ]

    import ollama
    response = ollama.chat(
        model=MODEL_NAME,
        options={
            "num_ctx": 16_384,
        },
        messages=messages
    )
    
    tool_call = convert_llm_string_output_to_tool(response["message"]["content"])
    try:
        pydantic_model(**tool_call["arguments"])
    except:
        return None
    return tool_call["arguments"]


MODEL_NAME = "llama3.1:8b"


if __name__ == "__main__":
    outputs = []
    for _ in tqdm(range(10)):
        outputs.append(
            test_asset_extraction()
        )
    exit()