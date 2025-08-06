from __future__ import annotations

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

# TODO Wrap the whole chatbot into its own service

ASSET_TYPES = [typ.value for typ in settings.AIOD.ASSET_TYPES]
MISTRAL_CLIENT = Mistral(api_key=settings.CHATBOT.MISTRAL_KEY)

# TODO replace this with the embedding store MilvusEmbeddingStore
MILVUS_CLIENT = MilvusClient(uri=str(settings.MILVUS.URI), token=settings.MILVUS.MILVUS_TOKEN)

EMBEDDING_MODEL = AiModel(device="cpu")

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
def start_conversation(user_query: str) -> tuple[str, str]:
    if moderate_input(user_query):
        response = MISTRAL_CLIENT.beta.conversations.start(agent_id=talk2aiod.id, inputs=user_query)
        result = handle_function_call(response)
        return result, response.conversation_id
    else:
        return "I can not answer this question.", "-1"


def continue_conversation(user_query: str, conversation_id: str) -> str:
    if moderate_input(user_query):
        response = MISTRAL_CLIENT.beta.conversations.append(
            conversation_id=conversation_id, inputs=user_query
        )
        result = handle_function_call(response)
        return result
    else:
        return "I can not answer this question."


def aiod_page_search(query: str) -> str:
    """Used to explain the AIoD website."""
    return get_relevant_crawled_content(query, settings.CHATBOT.WEBSITE_COLLECTION_NAME)


def aiod_api_search(query: str) -> str:
    """Used to explain the AIoD API."""
    return get_relevant_crawled_content(query, settings.CHATBOT.API_COLLECTION_NAME)


def asset_search(query: str, asset: str) -> str:
    """
    Used to search for resources a user needs.
    :param query: what to search for
    :param asset: the type of asset searched for.
    :return:
    """
    embedding_vector = EMBEDDING_MODEL.compute_query_embeddings(query)[0]
    mapped_asset = map_asset_name(asset)
    collection_to_search = settings.MILVUS.COLLECTION_PREFIX + "_" + mapped_asset

    docs = list(
        MILVUS_CLIENT.search(
            collection_name=collection_to_search,
            data=[embedding_vector],
            limit=3,
            group_by_field="asset_id",
            output_fields=["asset_id"],
            search_params={"metric_type": "COSINE"},
        )
    )[0]

    result = ""
    for doc in docs:
        content = get_aiod_asset(doc["entity"]["asset_id"], SupportedAssetType(mapped_asset))
        try:
            new_addition = (
                f"name: {content['name']}, publication date:{content['date_published']}, url: {content['same_as']}"  # type: ignore[index]
                f"\ncontent: {content['description']['plain']}\n"  # type: ignore[index]
            )
            result += new_addition
        except KeyError:
            pass

    return result


def get_relevant_crawled_content(query: str, collection_name: str) -> str:
    embedding_vector = EMBEDDING_MODEL.compute_query_embeddings(query)[0]

    docs = list(
        MILVUS_CLIENT.search(
            collection_name=collection_name,
            data=[embedding_vector],
            limit=3,
            output_fields=["url", "content"],
            search_params={"metric_type": "COSINE"},
        )
    )[0]

    result = ""
    for index, doc in enumerate(docs):
        result += (
            "Result "
            + str(index)
            + ":\n"
            + doc["entity"]["content"]
            + "\nlink:"
            + doc["entity"]["url"]
            + "\n"
        )
    return result


def map_asset_name(asset: str) -> str:
    """
    Map the asset named by the LLM to the closest existing asset using the edit distance.
    """
    lower_asset = asset.lower()
    dist_list = []
    for a in ASSET_TYPES:
        difference = edit_distance(lower_asset, a)
        dist_list.append(difference)

    smallest_dist_id = 0
    smallest_dist = dist_list[0]
    for i, dist in enumerate(dist_list):
        if dist < smallest_dist:
            smallest_dist_id = i

    return ASSET_TYPES[smallest_dist_id]


def handle_function_call(input_response: ConversationResponse) -> str:
    if input_response.outputs[-1].type == "function.call":
        if input_response.outputs[-1].name == "aiod_api_search":
            function_result = json.dumps(
                aiod_api_search(**json.loads(input_response.outputs[-1].arguments))
            )
        elif input_response.outputs[-1].name == "asset_search":
            function_result = json.dumps(
                asset_search(**json.loads(input_response.outputs[-1].arguments))
            )
        elif input_response.outputs[-1].name == "aiod_page_search":
            function_result = json.dumps(
                aiod_page_search(**json.loads(input_response.outputs[-1].arguments))
            )
        else:
            return input_response.outputs[-1].content  # no content in response outputs

        # Providing the result to our Agent
        user_function_calling_entry = FunctionResultEntry(
            tool_call_id=input_response.outputs[-1].tool_call_id,
            result=function_result,
        )

        # Retrieving the final response
        tool_response = MISTRAL_CLIENT.beta.conversations.append(
            conversation_id=input_response.conversation_id, inputs=[user_function_calling_entry]
        )
        return handle_function_call(tool_response)
    else:
        return input_response.outputs[-1].content


def moderate_input(input_query: str) -> bool:
    response = MISTRAL_CLIENT.classifiers.moderate_chat(
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
