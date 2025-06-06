from langchain_mistralai import ChatMistralAI
from mistralai import Mistral
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor, ZeroShotAgent
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain_core.tools import tool
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from time import sleep
from uuid import uuid4
from dotenv import load_dotenv
import sys
from datetime import datetime

from typing import Optional
import os

import torch
from pymilvus import MilvusClient, DataType
from app.services.inference.architecture import Basic_EmbeddingModel, SentenceTransformerToHF
# from app.services.aiod import perform_url_request, get_aiod_asset -> importing this produces a lot of errors
from nltk import edit_distance
from urllib.parse import urljoin
import requests

from prompt_library import *

# load chatbot .env
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

# load client for moderation
client = Mistral(api_key=mistral_key)

# load llm
llm = ChatMistralAI(
    model="ministral-8b-latest",  # "mistral-medium-latest",  #
    temperature=0,
    max_retries=2,
    mistral_api_key=mistral_key
)

# load db
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


class SearchInput(BaseModel):  # https://python.langchain.com/docs/modules/tools/custom_tools/
    query: str = Field(description="should be a search query")


@tool("website_search-tool", args_schema=SearchInput, return_direct=False)
def aiod_page_search(query: str) -> str:
    """Used to explain the AIoD website."""
    print("aiod_page_search:", str(datetime.now()), "query:", query)
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
    print("aiod_page_search output", result)
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


class AssetSearchInput(BaseModel):
    query: str = Field(description="the input query")
    asset: str = Field(description=f"the type of asset searched for. Options are {asset_types}")


@tool("asset_search-tool", args_schema=AssetSearchInput, return_direct=False)
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
    print(docs)

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
            print(content)
            pass

    print("asset_search output", result)
    return result


class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")


@tool("multiplication-tool", args_schema=CalculatorInput, return_direct=False)
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


# tools = [PageSearch(), AssetSearch(), multiply()]
tools = [multiply, asset_search, aiod_page_search]

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)

final_prompt = prompt

# print(final_prompt.template)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def reset_history(session_id):
    history = ChatMessageHistory(session_id=session_id)
    history.clear()


def add_outside_info_into_context(interests, publication_history):
    scholar_prefix = """You are an intelligent interactive assistant that manages the AI on Demand (AIoD) website. 
    The AIoD website consists of multiple parts. It has been created to facilitate collaboration, exchange and development of AI in Europe.
    For example, users can up/download pre-trained AI models, find/upload scientific publications and access/provide a number of datasets.
    It is your job to help the user navigate the website using the page_search or help the user find resources using the resource_search providing links to the websites/sources you are talking about.
    Always provide links to the resources and websites you talk about. If you cannot find information about something say: 'I have no information on this field, can you reformulate the question?'
    You are having a conversation with a person with the following interests: {interests}
    and publication history: {publication_history}
    You have access to the following tools:
    """.format(interests=interests, publication_history=publication_history)
    scholar_prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=scholar_prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )
    return scholar_prompt


def summarize_messages(session_identifier):
    history = ChatMessageHistory(session_id=uuid4())
    stored_messages = history.messages
    # print("STORED MESSAGES", stored_messages)
    if len(stored_messages) == 0:
        return False
    summarization_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "user",
                "Distill the above chat messages into a single summary message. Include as many specific details as you can.",
            ),
        ]
    )
    summarization_chain = summarization_prompt | llm
    try:  # try to summarize. If that fails keep working without summary
        summary_message = summarization_chain.invoke({"chat_history": stored_messages})
    except:
        summary_message = stored_messages
    history.clear()
    history.add_message(summary_message)
    # print("HISTORY", history.messages)

    return True


def repeatedly_invoke(agent, input_query, config):
    try:
        response = agent.invoke({'input': input_query}, config=config)
    except:
        sleep(2)
        try:
            response = agent.invoke({'input': input_query}, config=config)
        except:
            sleep(4)
            try:
                response = agent.invoke({'input': input_query}, config=config)
            except Exception as e:
                response = {"output": "Something went wrong, please try again."}
                print(datetime.now(), e)
    return response


def clean_final_response(final_response_str: str, words: list) -> str:
    if "Final Answer:" in final_response_str:
        return final_response_str.split("Final Answer:")[-1]
    else:
        lines = final_response_str.splitlines()
        filtered_lines = [line for line in lines if not any(line.startswith(word) for word in words)]
    return "\n".join(filtered_lines)


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


def agent_response(input_query, session_identifier, personalization=None):
    # print("session id", session_identifier)
    agent = create_openai_tools_agent(llm, tools, final_prompt)
    """if personalization:
        agent = create_openai_tools_agent(llm, tools, personalization)
    else:
        agent = create_openai_tools_agent(llm, tools, prompt_without_history)"""

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        max_iterations=6,
        max_execution_time=60,  # seconds use 1 second for exception testing
        # early_stopping_method="generate"
        # generate is not implemented on langchain-side even though its documentation says otherwise
    )
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,  # lambda session_id: ChatMessageHistory(session_id=session_identifier),
        input_messages_key='input',
        history_messages_key='chat_history'
    )

    print("input query:", input_query)
    answer_question = moderate_input(input_query)
    print("moderation result:", answer_question)
    if answer_question:
        response = repeatedly_invoke(agent_with_chat_history, input_query, config={'configurable': {"session_id": session_identifier}})
        print(response)
        final_response = response['output']
        # print("final_response:", final_response)

        if final_response == 'Agent stopped due to max iterations.':
            final_response = response['intermediate_steps']

        if isinstance(final_response, list):
            try:
                new_input = final_response[-1][1]
                exception_prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "Answer the given question to the best of your ability using the provided context. Make sure to cite your sources.",
                        ),
                        MessagesPlaceholder(variable_name="history"),
                        ("assistant", "Context: {context}"),
                        ("human", "{input}"),
                    ]
                )
                runnable = exception_prompt | llm
                with_message_history = RunnableWithMessageHistory(
                    runnable,
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="history",
                )
                final_response = with_message_history.invoke({"input": input_query, "context": new_input}, config={
                    'configurable': {"session_id": session_identifier}}).content

            except:
                final_response = 'I encountered an error. Could you reformulate the question?' + str(final_response)

        final_r = clean_final_response(final_response, ["Thought:", "Action:", "Action Input:", "Question:", "Observation:"])
    # clean_r = final_r.replace("Observation: ", "").replace("Final Answer: ", "")
    else:
        final_r = "I cannot answer this question."
    return final_r


# a = agent_response("how can I get in contact with the aiod community", -1)
a = agent_response("find machine learning models for image classification", -1)
# a = agent_response("multiply 500*12341345", -1)
print("###Agent result: ###")
print(a)

