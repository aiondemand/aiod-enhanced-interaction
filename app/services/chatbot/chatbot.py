from __future__ import annotations
import logging
import asyncio
from typing import Callable, Awaitable

import numpy as np
import json
from mistralai import Mistral, FunctionResultEntry
from mistralai.models.conversationresponse import ConversationResponse
from pymilvus import MilvusClient
from nltk import edit_distance

from app.schemas.enums import SupportedAssetType
from app.services.aiod import get_aiod_asset
from app.services.chatbot.prompt_library import *
from app.services.inference.model import AiModel
from app.config import settings

# Github issue: https://github.com/aiondemand/aiod-enhanced-interaction/issues/127
# TODO Wrap the whole chatbot into its own service
# TODO replace this with the embedding store MilvusEmbeddingStore

ASSET_TYPES = [typ.value for typ in settings.AIOD.ASSET_TYPES]
MISTRAL_CLIENT = Mistral(api_key=settings.CHATBOT.MISTRAL_KEY)
MILVUS_CLIENT = MilvusClient(uri=str(settings.MILVUS.URI), token=settings.MILVUS.MILVUS_TOKEN)
EMBEDDING_MODEL = AiModel(device="cpu")


# TOOLS EXPOSED TO THE AGENT
# TODO For the sake of simplicity and time, these tools are executed in a separate thread
# as milvus and model embedding computation are blocking calls
# TODO fix later once Milvus and other services are implemented in async way
async def aiod_page_search(query: str) -> str:
    """Used to explain the AIoD website."""
    logging.info(f"Tool 'aiod_page_search' has been called with a query: '{query}'")

    return await asyncio.to_thread(
        _get_relevant_crawled_content, query, settings.CHATBOT.WEBSITE_COLLECTION_NAME
    )


async def aiod_api_search(query: str) -> str:
    """Used to explain the AIoD API."""
    logging.info(f"Tool 'aiod_api_search' has been called with a query: '{query}'")
    return await asyncio.to_thread(
        _get_relevant_crawled_content, query, settings.CHATBOT.API_COLLECTION_NAME
    )


async def asset_search(query: str, asset: str) -> str:
    """
    Used to search for resources a user needs.
    :param query: what to search for
    :param asset: the type of asset searched for.
    :return:
    """
    logging.info(
        f"Tool 'asset_search' has been called with a query: '{query}' and an asset: '{asset}'"
    )
    return await asyncio.to_thread(_asset_search, query, asset)


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "asset_search",
            "description": "Used to search for resources a user needs",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for",
                    },
                    "asset": {
                        "type": "string",
                        "description": f"The type of asset searched for. Available options are {ASSET_TYPES}.",
                    },
                },
                "required": ["query", "asset"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aiod_page_search",
            "description": "Used to search for information on the AIoD website and to guide the user through it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aiod_api_search",
            "description": "Used to search for information on the api of AIoD and to guide the user through it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for",
                    },
                },
                "required": ["query"],
            },
        },
    },
]
TOOLS: dict[str, Callable[..., Awaitable[str]]] = {
    "aiod_api_search": aiod_api_search,
    "asset_search": asset_search,
    "aiod_page_search": aiod_page_search,
}


talk2aiod = MISTRAL_CLIENT.beta.agents.create(
    model=settings.CHATBOT.MISTRAL_MODEL,
    description="The Talk2AIoD chatbot",
    name="Talk2AIoD",
    tools=TOOL_DEFINITIONS,
    instructions=master_prompt,
    completion_args={
        "temperature": 0.3,
        "top_p": 0.95,
    },
    # tool_choice = "any",
    # parallel_tool_calls = True,
)


# TODO decide if we can use the mistral cloud to store conversations or have to build our own solution
async def start_conversation(user_query: str) -> tuple[str, str | None]:
    if await moderate_input(user_query):
        response = await MISTRAL_CLIENT.beta.conversations.start_async(
            agent_id=talk2aiod.id, inputs=user_query
        )
        result = await handle_function_call(response)
        return result, response.conversation_id
    else:
        return "I can not answer this question.", None


async def continue_conversation(user_query: str, conversation_id: str) -> str:
    if await moderate_input(user_query):
        response = await MISTRAL_CLIENT.beta.conversations.append_async(
            conversation_id=conversation_id, inputs=user_query
        )
        result = await handle_function_call(response)
        return result
    else:
        return "I can not answer this question."


async def get_conversation_messages(conversation_id: str):
    return await MISTRAL_CLIENT.beta.conversations.get_messages_async(
        conversation_id=conversation_id
    )


def _asset_search(query: str, asset: str) -> str:
    embedding_vector = EMBEDDING_MODEL.compute_query_embeddings(query)[0]
    mapped_asset = map_asset_name(asset)
    collection_to_search = settings.MILVUS.COLLECTION_PREFIX + "_" + mapped_asset

    result: str = ""
    satisfactory_docs = 0
    num_docs_to_return = settings.CHATBOT.TOP_K_ASSETS_TO_SEARCH
    seen_asset_ids: list[str] = []

    attempt = 0
    max_attempts = 5

    while satisfactory_docs < num_docs_to_return and attempt < max_attempts:
        filter_expr = None if len(seen_asset_ids) == 0 else f"asset_id not in {seen_asset_ids}"

        docs = list(
            MILVUS_CLIENT.search(
                collection_name=collection_to_search,
                data=[embedding_vector],
                limit=num_docs_to_return,
                group_by_field="asset_id",
                output_fields=["asset_id"],
                search_params={"metric_type": "COSINE"},
                filter=filter_expr,
            )
        )[0]
        if len(docs) == 0:
            break

        for doc in docs:
            asset_id = doc["entity"]["asset_id"]
            seen_asset_ids.append(asset_id)

            content = get_aiod_asset(asset_id, SupportedAssetType(mapped_asset))
            if content is None:
                continue

            try:
                new_addition = (
                    f"name: {content['name']}, publication date:{content['date_published']}, url: {content['same_as']}"  # type: ignore[index]
                    f"\ncontent: {content['description']['plain']}\n"  # type: ignore[index]
                )
                result += new_addition
                satisfactory_docs += 1
                if satisfactory_docs == settings.CHATBOT.TOP_K_ASSETS_TO_SEARCH:
                    break
            except KeyError:
                continue
        attempt += 1

    return result


def _get_relevant_crawled_content(query: str, collection_name: str) -> str:
    embedding_vector = EMBEDDING_MODEL.compute_query_embeddings(query)[0]

    docs = list(
        MILVUS_CLIENT.search(
            collection_name=collection_name,
            data=[embedding_vector],
            limit=settings.CHATBOT.TOP_K_ASSETS_TO_SEARCH,
            output_fields=["url", "content"],
            search_params={"metric_type": "COSINE"},
        )
    )[0]

    return "".join(
        f"Result {index}:\n{doc['entity']['content']}\nlink:{doc['entity']['url']}\n"
        for index, doc in enumerate(docs)
    )


def map_asset_name(asset: str) -> str:
    """
    Map the asset named by the LLM to the closest existing asset using the edit distance.
    """
    dist_list = np.array([edit_distance(asset.lower(), a) for a in ASSET_TYPES])
    return ASSET_TYPES[int(np.argmin(dist_list))]


async def handle_function_call(input_response: ConversationResponse) -> str:
    if input_response.outputs[-1].type == "function.call":
        function_args = json.loads(input_response.outputs[-1].arguments)
        function_name = input_response.outputs[-1].name

        if function_name in TOOLS.keys():
            function_result = json.dumps(await TOOLS[function_name](**function_args))
        else:
            return input_response.outputs[-1].content

        # Providing the result to our Agent
        user_function_calling_entry = FunctionResultEntry(
            tool_call_id=input_response.outputs[-1].tool_call_id,
            result=function_result,
        )
        # Retrieving the final response
        tool_response = await MISTRAL_CLIENT.beta.conversations.append_async(
            conversation_id=input_response.conversation_id, inputs=[user_function_calling_entry]
        )
        return await handle_function_call(tool_response)
    else:
        return input_response.outputs[-1].content


async def moderate_input(input_query: str) -> bool:
    response = await MISTRAL_CLIENT.classifiers.moderate_chat_async(
        model="mistral-moderation-latest",
        inputs=[
            {"role": "user", "content": input_query},
            {"role": "assistant", "content": "...assistant response..."},
        ],
    )
    for category in response.results[0].categories.keys():
        if response.results[0].categories[category]:
            return False
    return True
