from langchain_mistralai import ChatMistralAI
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor, ZeroShotAgent
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
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

# load llm
llm = ChatMistralAI(
    model="ministral-8b-latest",
    temperature=0,
    max_retries=2,
    mistral_api_key=mistral_key
)

# load db
milvus_db = MilvusClient(uri=milvus_uri, token=milvus_token)


def prepare_embedding_model() -> Basic_EmbeddingModel:
    transformer = SentenceTransformerToHF(embedding_llm, trust_remote_code=True)
    if torch.cuda.is_available() and use_gpu:
            # print("use cuda")
        transformer.cuda()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
            # print("use cpu")
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


# https://python.langchain.com/docs/modules/memory/agent_with_memory/
# modified memory to summarize previous messages ensure context length doesn't get out of control
#  possibilities: only store last n messages, summarize conversation history
#  https://python.langchain.com/docs/use_cases/chatbots/memory_management/
# https://python.langchain.com/docs/use_cases/tool_use/prompting/
# https://python.langchain.com/docs/use_cases/tool_use/multiple_tools/
# https://python.langchain.com/docs/modules/agents/
# https://python.langchain.com/docs/langgraph/
# https://python.langchain.com/docs/use_cases/sql/agents/
# https://python.langchain.com/docs/use_cases/sql/
# https://python.langchain.com/docs/modules/tools/
# store = {'-1': ChatMessageHistory(messages=[HumanMessage(content='mushroom dataset'), AIMessage(content="Question: mushroom dataset\nThought: I should use the resource_search tool to find datasets related to mushrooms.\nAction: resource_search\nAction Input: mushroom dataset\nObservation: \ndatasets 1:\nhttps://www.openml.org/search?type=data&id=43922\nmushroom\n['machine learning', 'meteorology']\nMushroom records drawn from The Audubon Society Field Guide to North American Mushrooms (1981). G. H. Lincoff (Pres.), New York: Alfred A. Knopf\nlink:https://www.openml.org/search?type=data&id=43922\n\ndatasets 2:\nhttps://www.openml.org/search?type=data&id=43923\nmushroom\n['machine learning', 'medicine']\nNursery Database was derived from a hierarchical decision model originally developed to rank applications for nursery schools.\nlink:https://www.openml.org/search?type=data&id=43923\n\ndatasets 3:\nhttps://zenodo.org/api/records/8212067\nA new species of smooth-spored Inocybe from Coniferous forests of Pakistan\n['taxonomy', 'mushroom', 'mycology', 'inocybe', 'pakistan']\nInocybe bhurbanensis is described and illustrated as a new species from Himalayan Moist Temperate forests of Pakistan. It is characterized by fibrillose, conical to convex, umbonate, brown to dark brown pileus, non-pruinose, fibrillose stipe with whitish tomentum at the base and smooth basidiospores that are larger (9 × 5.2 µm) and thicker caulocystidia (up to 21 m) as compared to the sister species Inocybe demetris. Phylogenetic analyses of a nuclear rDNA region encompassing the internal transcribed spacers 1 and 2 along with 5.8S rDNA (ITS) and the 28S rDNA D1–D2 domains (28S) also confirmed its novelty.\nlink:https://zenodo.org/api/records/8212067\n\ndatasets 4:\nhttps://zenodo.org/api/records/7797389\nData from: Effects of fungicides on aquatic fungi and bacteria: a comparison of morphological and molecular approaches from a microcosm experiment\n['dna metabarcoding', 'streams', 'community composition', 'stress response', 'leaf decomposition']\nData files and R code for the manuscript: Effects of fungicides on aquatic fungi and bacteria: a comparison of morphological and molecular approaches from a microcosm experiment. Published in Environmental Sciences Europe.\nlink:https://zenodo.org/api/records/7797389\n\ndatasets 5:\nhttps://zenodo.org/api/records/8210711\nSupplementary material 4 from: Jung P, Werner L, Briegel-Williams L, Emrich D, Lakatos M (2023) Roccellinastrum, Cenozosia and Heterodermia: Ecology and phylogeny of fog lichens and their photobionts from the coastal Atacama Desert. MycoKeys 98: 317- [...]\n['niebla', 'symbiochloris', 'pan de azucar', 'heterodermia', 'chlorolichens', 'trebouxia']\nSpot tests and TLC\nlink:https://zenodo.org/api/records/8210711\n\nThought: I now know the final answer\nFinal Answer: Here are some datasets related to mushrooms:\n\n1. [Mushroom records from The Audubon Society Field Guide to North American Mushrooms (1981)](https://www.openml.org/search?type=data&id=43922)\n2. [Nursery Database derived from a hierarchical decision model](https://www.openml.org/search?type=data&id=43923)\n3. [A new species of smooth-spored Inocybe from Coniferous forests of Pakistan](https://zenodo.org/api/records/8212067)\n4. [Effects of fungicides on aquatic fungi and bacteria](https://zenodo.org/api/records/7797389)\n5. [Supplementary material on fog lichens and their photobionts from the coastal Atacama Desert](https://zenodo.org/api/records/8210711)")])}
store = {}
embedding_model = prepare_embedding_model()


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def reset_history(session_id):
    history = ChatMessageHistory(session_id=session_id)
    history.clear()


class SearchInput(BaseModel):  # https://python.langchain.com/docs/modules/tools/custom_tools/
    query: str = Field(description="should be a search query")


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
    # print("tool output", result)
    return result


def map_asset(asset: str) -> str:
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
    url = urljoin(str(aiod_url), f"{asset}/v1/{str(asset_id)}")
    # print(url)
    result = requests.get(url)
    return result.json()


def asset_search(query: str, asset: str) -> str:
    """
        Used to search for resources a user needs.
        :param query: user input
        :param asset: the type of asset searched for from
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
        new_addition = f'name: {content["name"]}, publication date:{ content["date_published"]}, url: {content["same_as"]}' \
                       f'\ncontent: {content["description"]["plain"]}\n'
        # print(content)
        result += new_addition

    print(result)
    return result


asset_search("find me a mushroom dataset", "datasets")
ps_desc = """Use the unmodified user input to get information about specific webpages on the AIoD website."""


class PageSimilaritySearch(BaseTool):
    name: str = "page_search"
    description: str = ps_desc
    # args_schema: Type[BaseModel] = SearchInput

    def _run(
        self,
        query: str,
    ) -> str: return aiod_page_search(query)

    async def _arun(
        self,
        query: str,
        data_type: Optional[str] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


rsd_desc = f"Use this to search for assets that fit to the users requests. Available asset types are: {asset_types}."

rsd_desc_kw = """Use this to search for specific user requests in the resources. Searches for documents of the specified resource type where all words in the query are in the document. Available resource types are: publications, datasets, educational_resources, experiments, ml_models.
    """
rsd_args = """Example: user: I want to know something about drones
    Action: use resource_search
    Result:
    publications 1: Citizen Consultation | Drone-in-a-box by NAUST Robotics keywords: Robotics4EU, Citizen Consultation, Drone-in-a-box, NAUST Robotics, Publication
The HTML code contains metadata and links for a website related to Robotics4EU, focusing on citizen consultation and drone technology by NAUST Robotics. The webpage includes social media tags and structured data for SEO optimization.
link:https://www.robotics4eu.eu/publications/citizen-consultation-drone-in-a-box-by-naust-robotics/"""

rsd_desc_v2 = """Use this tool search for resources that fit to the users requests. Available resource types are: publications, datasets, educational_resources, experiments, ml_models. If you want to search for multiple resources at the same time, use keyword 'all'. 
    Make sure to give resource_search an argument structured like this: \{'input': 'the thing you want to search for', 'type': 'the type of the thing you want to search'\}
    Example: user: show me publications about drones
    Action: use resource_search with the input \{'input': 'drones', 'type': 'publications'\}
    Result:
    publications 1: Citizen Consultation | Drone-in-a-box by NAUST Robotics keywords: Robotics4EU, Citizen Consultation, Drone-in-a-box, NAUST Robotics, Publication
The HTML code contains metadata and links for a website related to Robotics4EU, focusing on citizen consultation and drone technology by NAUST Robotics. The webpage includes social media tags and structured data for SEO optimization.
link:https://www.robotics4eu.eu/publications/citizen-consultation-drone-in-a-box-by-naust-robotics/"""


class ResourceSimilaritySearch(BaseTool):
    name: str = "resource_similarity_search"
    description: str = rsd_desc
    # args_schema: Type[BaseModel] = SearchInput

    def _run(
        self,
        query: str,
        asset: str,
    ) -> str: return asset_search(query, asset)

    async def _arun(
        self,
        query: str,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")


tools = [PageSimilaritySearch(), ResourceSimilaritySearch()]



prefix = """You are an intelligent interactive assistant that manages the AI on Demand (AIoD) website. It is an amalgamation consisting of multiple parts. For example, you can up/download pre-trained AI models, find/upload scientific publications and access/provide a number of relevant datasets.

Your goal is to guide new users, that have not visited the website you manage, to the resources they want. 
To guide users you will first have to gather some information on the user.
1. You have to figure out what interests the user has that overlap with the content provided on the website
2. You have to figure out what the user wants to get from the website

You can ask the users questions as you see fit, but make sure to stay on topic and stop the questioning once you gathered sufficient amount of information to guide the user towards an initial goal or after asking 3 questions. If the user wants more help afterwards, make sure to provide it and keep asking questions where and when needed. 

To help the user navigate the website, you can provide links to pages you deem relevant. Always provide links to the ressources you talk about. You have access to the following tools:"""

prefix2 = """You are an intelligent interactive assistant that manages the AI on Demand (AIoD) website. 
The AIoD website consists of multiple parts. It has been created to facilitate collaboration, exchange and development of AI in Europe.
For example, users can up/download pre-trained AI models, find/upload scientific publications and access/provide a number of datasets.
It is your job to help the user navigate the website using the page_search or help the user find resources using the resource_search providing links to the websites/sources you are talking about.
Always provide links to the resources and websites you talk about. After your search, check carefully if the results contain the information you need to answer the question. If you cannot find the information you are searching for, reformulate the query by removing stop words or using synonyms.
Only if you have exhausted all other options, say: 'I found no results answering your question, can you reformulate it?'
You have access to the following tools:
"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

suffix_v2 = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix2,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)

final_prompt = prompt

# print(final_prompt.template)


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
    # if "Observation: " in final_response_str and "Final Answer: " in final_response_str:
    #    words.append("Observation: ")
    lines = final_response_str.splitlines()
    filtered_lines = [line for line in lines if not any(line.startswith(word) for word in words)]
    return "\n".join(filtered_lines)


def clean_final_response_v2(final_response_str: str, words: list) -> str:
    if "Final Answer:" in final_response_str:
        return final_response_str.split("Final Answer:")[-1]
    else:
        lines = final_response_str.splitlines()
        filtered_lines = [line for line in lines if not any(line.startswith(word) for word in words)]
    return "\n".join(filtered_lines)


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

    print("input query", input_query)
    # print("session id", session_identifier)
    # response = agent_with_chat_history.invoke({'input': input_query}, config={'configurable': {"session_id": session_identifier}})
    response = repeatedly_invoke(agent_with_chat_history, input_query, config={'configurable': {"session_id": session_identifier}})
    """try:
        response = agent_with_chat_history.invoke({'input': input_query},
                                                   config={'configurable': {"session_id": session_identifier}})
    except Exception as e:
        response = {"output": "Something went wrong, please try again."}
        print(datetime.now(), e)"""

    final_response = response['output']  # .split("Final Answer:")[-1]

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

    final_r = clean_final_response_v2(final_response, ["Thought:", "Action:", "Action Input:", "Question:", "Observation:"])
    # clean_r = final_r.replace("Observation: ", "").replace("Final Answer: ", "")

    return final_r

