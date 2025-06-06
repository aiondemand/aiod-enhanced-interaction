import json
import functools
import os
from mistralai import Mistral, FunctionResultEntry
from dotenv import load_dotenv
import torch
from pymilvus import MilvusClient, DataType
from app.services.inference.architecture import Basic_EmbeddingModel, SentenceTransformerToHF
# from app.services.aiod import perform_url_request, get_aiod_asset -> importing this produces a lot of errors
from nltk import edit_distance
from urllib.parse import urljoin
import requests
from datetime import datetime
from app.chatbot.prompt_library import *

load_dotenv(".env.chatbot")
mistral_key = os.getenv("MISTRAL_KEY")
website_collection = os.getenv("WEBSITE_COLLECTION")

# load app .env
load_dotenv("../../.env.app")
milvus_uri = os.getenv("MILVUS__URI")
milvus_token = os.getenv('MILVUS__USER')+":"+os.getenv('MILVUS__PASS')
embedding_llm = os.getenv('MODEL_LOADPATH')
use_gpu = os.getenv('USE_GPU')
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


# store = {'-1': ChatMessageHistory(messages=[HumanMessage(content='mushroom dataset'), AIMessage(content="Question: mushroom dataset\nThought: I should use the resource_search tool to find datasets related to mushrooms.\nAction: resource_search\nAction Input: mushroom dataset\nObservation: \ndatasets 1:\nhttps://www.openml.org/search?type=data&id=43922\nmushroom\n['machine learning', 'meteorology']\nMushroom records drawn from The Audubon Society Field Guide to North American Mushrooms (1981). G. H. Lincoff (Pres.), New York: Alfred A. Knopf\nlink:https://www.openml.org/search?type=data&id=43922\n\ndatasets 2:\nhttps://www.openml.org/search?type=data&id=43923\nmushroom\n['machine learning', 'medicine']\nNursery Database was derived from a hierarchical decision model originally developed to rank applications for nursery schools.\nlink:https://www.openml.org/search?type=data&id=43923\n\ndatasets 3:\nhttps://zenodo.org/api/records/8212067\nA new species of smooth-spored Inocybe from Coniferous forests of Pakistan\n['taxonomy', 'mushroom', 'mycology', 'inocybe', 'pakistan']\nInocybe bhurbanensis is described and illustrated as a new species from Himalayan Moist Temperate forests of Pakistan. It is characterized by fibrillose, conical to convex, umbonate, brown to dark brown pileus, non-pruinose, fibrillose stipe with whitish tomentum at the base and smooth basidiospores that are larger (9 × 5.2 µm) and thicker caulocystidia (up to 21 m) as compared to the sister species Inocybe demetris. Phylogenetic analyses of a nuclear rDNA region encompassing the internal transcribed spacers 1 and 2 along with 5.8S rDNA (ITS) and the 28S rDNA D1–D2 domains (28S) also confirmed its novelty.\nlink:https://zenodo.org/api/records/8212067\n\ndatasets 4:\nhttps://zenodo.org/api/records/7797389\nData from: Effects of fungicides on aquatic fungi and bacteria: a comparison of morphological and molecular approaches from a microcosm experiment\n['dna metabarcoding', 'streams', 'community composition', 'stress response', 'leaf decomposition']\nData files and R code for the manuscript: Effects of fungicides on aquatic fungi and bacteria: a comparison of morphological and molecular approaches from a microcosm experiment. Published in Environmental Sciences Europe.\nlink:https://zenodo.org/api/records/7797389\n\ndatasets 5:\nhttps://zenodo.org/api/records/8210711\nSupplementary material 4 from: Jung P, Werner L, Briegel-Williams L, Emrich D, Lakatos M (2023) Roccellinastrum, Cenozosia and Heterodermia: Ecology and phylogeny of fog lichens and their photobionts from the coastal Atacama Desert. MycoKeys 98: 317- [...]\n['niebla', 'symbiochloris', 'pan de azucar', 'heterodermia', 'chlorolichens', 'trebouxia']\nSpot tests and TLC\nlink:https://zenodo.org/api/records/8210711\n\nThought: I now know the final answer\nFinal Answer: Here are some datasets related to mushrooms:\n\n1. [Mushroom records from The Audubon Society Field Guide to North American Mushrooms (1981)](https://www.openml.org/search?type=data&id=43922)\n2. [Nursery Database derived from a hierarchical decision model](https://www.openml.org/search?type=data&id=43923)\n3. [A new species of smooth-spored Inocybe from Coniferous forests of Pakistan](https://zenodo.org/api/records/8212067)\n4. [Effects of fungicides on aquatic fungi and bacteria](https://zenodo.org/api/records/7797389)\n5. [Supplementary material on fog lichens and their photobionts from the coastal Atacama Desert](https://zenodo.org/api/records/8210711)")])}
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
        limit=3
    )
    # print(docs[0])
    result = ""
    for index, doc in enumerate(docs):
        result += "Result "+str(index) + ":\n" + doc['content'] + "\nlink:" + doc['url'] + "\n"
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


def multiply(a: int, b: int) -> str:
    """Multiply two numbers."""
    return json.dumps({'result': str(a * b)})


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
    collection_to_search = collection_prefix+"_"+mapped_asset
    # print(collection_to_search)

    docs = milvus_db.query(
        collection_name=collection_to_search,
        anns_field=[embedding_vector],
        output_fields=["id", "asset_id"],
        limit=3
    )
    # print(docs)

    result = ""
    for index, doc in enumerate(docs):
        content = get_asset_from_aiod(doc['asset_id'], mapped_asset)
        # print(content)
        try:
            new_addition = f'name: {content["name"]}, publication date:{ content["date_published"]}, url: {content["same_as"]}' \
                       f'\ncontent: {content["description"]["plain"]}\n'
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
            "name": "multiply",
            "description": "Multiply two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "integer",
                        "description": "The first number to multiply.",
                    },
                    "b": {
                        "type": "integer",
                        "description": "The second number to multiply.",
                    }
                },
                "required": ["a", "b"],
            },
        },
    },
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
                    }
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
    }
]

names_to_functions = {
    'multiply': functools.partial(multiply),
    'asset_search': functools.partial(asset_search),
    'aiod_page_search': functools.partial(aiod_page_search)
}


def handle_function_call(input_response):
    if input_response.outputs[-1].type == "function.call":
        print("function.call")
        if input_response.outputs[-1].name == "multiply":
            print("multiply")
            function_result = json.dumps(multiply(**json.loads(input_response.outputs[-1].arguments)))
        elif input_response.outputs[-1].name == "asset_search":
            print('asset_search')
            function_result = json.dumps(asset_search(**json.loads(input_response.outputs[-1].arguments)))
        elif input_response.outputs[-1].name == "aiod_page_search":
            print("aiod_page_search")
            function_result = json.dumps(aiod_page_search(**json.loads(input_response.outputs[-1].arguments)))
        else:
            print("return 1", input_response.outputs[-1].content)
            return input_response.outputs[-1].content

        # print("function result", function_result)
        # Providing the result to our Agent
        user_function_calling_entry = FunctionResultEntry(
            tool_call_id=input_response.outputs[-1].tool_call_id,
            result=function_result,
        )

        # Retrieving the final response
        # print(user_function_calling_entry)
        tool_response = client.beta.conversations.append(
            conversation_id=input_response.conversation_id,
            inputs=[user_function_calling_entry]
        )
        print("tool response", tool_response)
        return handle_function_call(tool_response)
    else:
        print("return 2", input_response.outputs[-1].content)  # TODO result is not always correctly returned through the recursions
        return input_response.outputs[-1].content


talk2aiod = client.beta.agents.create(
    model="mistral-medium-latest",
    description="The Talk2AIoD chatbot",
    name="Talk2AIoD",
    tools=tools,
    instructions=prefix
    # tool_choice = "any",
    # parallel_tool_calls = True,
)


def start_conversation(user_query: str):
    response = client.beta.conversations.start(
        agent_id=talk2aiod.id,
        inputs=user_query
        )
    result = handle_function_call(response)
    return result, response.conversation_id


a = start_conversation("How can I contact AIoD")
print("aaaaaaaaa", a[0], a[1])


def continue_conversation(user_query: str, conversation_id: str):
    response = client.beta.conversations.append(
        conversation_id=conversation_id, inputs=user_query
    )
    print(response)
    result = handle_function_call(response)
    return result


b = continue_conversation("who is jennifer renoux", a[1])
