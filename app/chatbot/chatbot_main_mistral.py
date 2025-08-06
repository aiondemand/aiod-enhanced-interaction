import json
import functools
import os
from mistralai import Mistral, FunctionResultEntry
from dotenv import load_dotenv
import torch
from pymilvus import MilvusClient
from app.services.inference.architecture import Basic_EmbeddingModel, SentenceTransformerToHF

# from app.services.aiod import perform_url_request, get_aiod_asset -> importing this produces a lot of errors
from nltk import edit_distance
from urllib.parse import urljoin
import requests
from app.chatbot.prompt_library import *

# load_dotenv(".env.chatbot")   # run directly
load_dotenv("app/chatbot/.env.chatbot")  # run through chatbot endpoint
mistral_key = os.getenv("MISTRAL_KEY")
website_collection = os.getenv("WEBSITE_COLLECTION")
api_collection = os.getenv("API_COLLECTION")

# load app .env
# load_dotenv("../../.env.app")  # run directly
load_dotenv(".env.app")  # run through chatbot endpoint
milvus_uri = os.getenv("MILVUS__URI")
milvus_token = os.getenv("MILVUS__USER") + ":" + os.getenv("MILVUS__PASS")
embedding_llm = os.getenv("MODEL_LOADPATH")
use_gpu = os.getenv("USE_GPU")
asset_types = os.getenv("AIOD__COMMA_SEPARATED_ASSET_TYPES").split(",")
collection_prefix = os.getenv("MILVUS__COLLECTION_PREFIX")
aiod_url = os.getenv("AIOD__URL")

client = Mistral(api_key=mistral_key)
model = "mistral-medium-latest"

milvus_db = MilvusClient(uri=milvus_uri, token=milvus_token)


def prepare_embedding_model() -> Basic_EmbeddingModel:
    transformer = SentenceTransformerToHF(embedding_llm, trust_remote_code=True)
    if torch.cuda.is_available() and use_gpu:
        transformer.cuda()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    em_model = Basic_EmbeddingModel(
        transformer,
        transformer.tokenizer,
        pooling="none",
        document_max_length=4096,
        dev=device,
    )
    return em_model


@torch.no_grad()
def embed_query(model, query: str) -> list[list[float]]:
    embedding = model.forward(query)[0].cpu().numpy().tolist()
    return embedding


store = {}
embedding_model = prepare_embedding_model()


def aiod_page_search(query: str) -> str:
    """Used to explain the AIoD website."""
    # print("aiod_page_search:", str(datetime.now()), "query:", query)
    embedding_vector = embed_query(embedding_model, query)
    docs = milvus_db.query(
        collection_name=website_collection,
        anns_field=[embedding_vector],
        output_fields=["url", "content"],
        limit=3,
    )
    # print(docs[0])
    result = ""
    for index, doc in enumerate(docs):
        result += "Result " + str(index) + ":\n" + doc["content"] + "\nlink:" + doc["url"] + "\n"
    # print("aiod_page_search output", result)
    return result


def aiod_api_search(query: str) -> str:
    """Used to explain the AIoD API."""
    embedding_vector = embed_query(embedding_model, query)
    docs = milvus_db.query(
        collection_name=api_collection,
        anns_field=[embedding_vector],
        output_fields=["url", "content"],
        limit=3,
    )
    # print(docs[0])
    result = ""
    for index, doc in enumerate(docs):
        result += "Result " + str(index) + ":\n" + doc["content"] + "\nlink:" + doc["url"] + "\n"
    # print("aiod_page_search output", result)
    return result


def map_asset(asset: str) -> str:
    """
    Map the asset named by the LLM to the closest existing asset using the edit distance.
    """
    lower_asset = asset.lower()
    dist_list = []
    for a in asset_types:
        difference = edit_distance(lower_asset, a)
        dist_list.append(difference)

    smallest_dist_id = 0
    smallest_dist = dist_list[0]
    for i, dist in enumerate(dist_list):
        if dist < smallest_dist:
            smallest_dist_id = i

    return asset_types[smallest_dist_id]


def get_asset_from_aiod(asset_id: int, asset: str):
    """
    Load content from metadata database for the asset with id asset_id.
    """
    url = urljoin(str(aiod_url), f"/v2/{asset}/{str(asset_id)}")
    result = requests.get(url)
    return result.json()


def asset_search(query: str, asset: str) -> str:
    """
    Used to search for resources a user needs.
    :param query: what to search for
    :param asset: the type of asset searched for.
    :return:
    """
    embedding_vector = embed_query(embedding_model, query)
    mapped_asset = map_asset(asset)
    # print("mapped_asset: ", mapped_asset)
    collection_to_search = collection_prefix + "_" + mapped_asset
    # print(collection_to_search)

    docs = milvus_db.query(
        collection_name=collection_to_search,
        anns_field=[embedding_vector],
        output_fields=["id", "asset_id"],
        limit=3,
    )
    # print(docs)

    result = ""
    for index, doc in enumerate(docs):
        content = get_asset_from_aiod(doc["asset_id"], mapped_asset)
        # print(content)
        try:
            new_addition = (
                f"name: {content['name']}, publication date:{content['date_published']}, url: {content['same_as']}"
                f"\ncontent: {content['description']['plain']}\n"
            )
            # print(content)
            result += new_addition
        except KeyError:
            # print(content)
            pass

    # print("asset_search output", result)
    return result


tools = [
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
                        "description": f"The type of asset searched for. Available options are {asset_types}.",
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
            "description": "Used to search for information on the website and to guide the user through it.",
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

names_to_functions = {
    "asset_search": functools.partial(asset_search),
    "aiod_page_search": functools.partial(aiod_page_search),
    "aiod_api_search": functools.partial(aiod_api_search),
}


def handle_function_call(input_response):
    print("handle_function_call", input_response)
    if input_response.outputs[-1].type == "function.call":
        print("function.call")
        if input_response.outputs[-1].name == "aiod_api_search":
            print("aiod_api_search", input_response.outputs[-1].arguments)
            function_result = json.dumps(
                aiod_api_search(**json.loads(input_response.outputs[-1].arguments))
            )
        elif input_response.outputs[-1].name == "asset_search":
            print("asset_search", input_response.outputs[-1].arguments)
            function_result = json.dumps(
                asset_search(**json.loads(input_response.outputs[-1].arguments))
            )
        elif input_response.outputs[-1].name == "aiod_page_search":
            print("aiod_page_search", input_response.outputs[-1].arguments)
            function_result = json.dumps(
                aiod_page_search(**json.loads(input_response.outputs[-1].arguments))
            )
        else:
            print("return 1", input_response.outputs[-1])
            return input_response.outputs[-1].content  # no content in response outputs

        # print("function result", function_result)
        # Providing the result to our Agent
        user_function_calling_entry = FunctionResultEntry(
            tool_call_id=input_response.outputs[-1].tool_call_id,
            result=function_result,
        )

        # Retrieving the final response
        # print(user_function_calling_entry)
        tool_response = client.beta.conversations.append(
            conversation_id=input_response.conversation_id, inputs=[user_function_calling_entry]
        )
        # print("tool response", tool_response)
        return handle_function_call(tool_response)
    else:
        return input_response.outputs[-1].content


def moderate_input(input_query, conversation=None):
    response = client.classifiers.moderate_chat(
        model="mistral-moderation-latest",
        inputs=[
            {"role": "user", "content": input_query},
            {"role": "assistant", "content": "...assistant response..."},
        ],
    )
    respond = True
    for category in response.results[0].categories.keys():
        if response.results[0].categories[category]:
            respond = False

    return respond


talk2aiod = client.beta.agents.create(
    model="mistral-medium-latest",
    description="The Talk2AIoD chatbot",
    name="Talk2AIoD",
    tools=tools,
    instructions=master_prompt,
    completion_args={
        "temperature": 0.3,
        "top_p": 0.95,
    },
    # tool_choice = "any",
    # parallel_tool_calls = True,
)


# TODO decide if we can use the mistral cloud to store conversations or have to build our own solution
def start_conversation(user_query: str):
    response = client.beta.conversations.start(agent_id=talk2aiod.id, inputs=user_query)
    if moderate_input(user_query):
        result = handle_function_call(response)
        return result, response.conversation_id
    else:
        return "I can not answer this question."


def continue_conversation(user_query: str, conversation_id: str):
    response = client.beta.conversations.append(conversation_id=conversation_id, inputs=user_query)
    if moderate_input(user_query):
        result = handle_function_call(response)
        return result
    else:
        return "I can not answer this question."


# a = start_conversation("How can I contact AIoD")
# print("start conversation", a[0], a[1])


# b = continue_conversation("who is jennifer renoux", a[1])
# print("continue conversation", b)

# c = continue_conversation("what did I ask for previously?", a[1])
# print("continue conversation2", c)
