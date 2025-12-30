from __future__ import annotations
import logging
from typing import Callable

import numpy as np
import json
from mistralai import ConversationMessages, Mistral, FunctionResultEntry
from mistralai.models.conversationresponse import ConversationResponse
from pymilvus import MilvusClient
from nltk import edit_distance


from app.schemas.enums import SupportedAssetType
from app.services.aiod import get_aiod_asset
from app.services.chatbot.chatbot_system_prompt import CHATBOT_SYSTEM_PROMPT
from app.services.inference.model import AiModel
from app import settings


ASSET_TYPES = [typ.value for typ in settings.AIOD.ASSET_TYPES]


class ChatbotTools:
    def __init__(self, milvus_client: MilvusClient, embedding_model: AiModel):
        self.milvus_client = milvus_client
        self.embedding_model = embedding_model

    def aiod_page_search(self, query: str) -> str:
        """Used to explain the AIoD website."""
        logging.info(f"Tool 'aiod_page_search' has been called with a query: '{query}'")
        return self._get_relevant_crawled_content(query, settings.CHATBOT.WEBSITE_COLLECTION_NAME)

    def aiod_api_search(self, query: str) -> str:
        """Used to explain the AIoD API."""
        logging.info(f"Tool 'aiod_api_search' has been called with a query: '{query}'")
        return self._get_relevant_crawled_content(query, settings.CHATBOT.API_COLLECTION_NAME)

    def asset_search(self, query: str, asset: str) -> str:
        """
        Used to search for resources a user needs.
        :param query: what to search for
        :param asset: the type of asset searched for.
        :return:
        """
        logging.info(
            f"Tool 'asset_search' has been called with a query: '{query}' and an asset: '{asset}'"
        )
        return self._asset_search(query, asset)

    def _asset_search(self, query: str, asset: str) -> str:
        """Internal method for asset search using the class's dependencies."""
        embedding_vector = self.embedding_model.compute_query_embeddings(query)[0]
        mapped_asset = self._map_asset_name(asset)
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
                self.milvus_client.search(
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
                    url = settings.CHATBOT.generate_mylibrary_asset_url(
                        asset_id, SupportedAssetType(mapped_asset)
                    )
                    if url is None:
                        url = content["same_as"]
                    new_addition = (
                        f"name: {content['name']}, publication date:{content['date_published']}, url: {url}"  # type: ignore[index]
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

    def _get_relevant_crawled_content(self, query: str, collection_name: str) -> str:
        """Internal method for crawled content search using the class's dependencies."""
        embedding_vector = self.embedding_model.compute_query_embeddings(query)[0]

        docs = list(
            self.milvus_client.search(
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

    def _map_asset_name(self, asset: str) -> str:
        """
        Map the asset named by the LLM to the closest existing asset using the edit distance.
        """
        dist_list = np.array([edit_distance(asset.lower(), a) for a in ASSET_TYPES])
        return ASSET_TYPES[int(np.argmin(dist_list))]

    def get_tools_dict(self) -> dict[str, Callable[..., str]]:
        """Get a dictionary of tool functions bound to this instance."""
        return {
            "aiod_api_search": self.aiod_api_search,
            "asset_search": self.asset_search,
            "aiod_page_search": self.aiod_page_search,
        }

    @classmethod
    def get_tool_definitions(cls) -> list[dict]:
        return [
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


class ChatbotService:
    """
    Service class encapsulating all chatbot functionality.
    Manages the Mistral agent, tools, and conversation logic.
    """

    def __init__(
        self,
        mistral_client: Mistral,
        milvus_client: MilvusClient,
        embedding_model: AiModel,
    ):
        """
        Initialize the chatbot service.

        Args:
            mistral_client: Mistral API client
            milvus_client: Milvus client for vector search
            embedding_model: AI model for embeddings
        """
        self.mistral_client = mistral_client
        self.tools_instance = ChatbotTools(milvus_client, embedding_model)
        self.tools = self.tools_instance.get_tools_dict()

        # Create the Mistral agent
        self.agent = self.mistral_client.beta.agents.create(
            model=settings.CHATBOT.MISTRAL_MODEL,
            description="AI assistant for AI-on-Demand platform",
            name="AI assistant",
            tools=self.tools_instance.get_tool_definitions(),
            instructions=CHATBOT_SYSTEM_PROMPT,
            completion_args={
                "temperature": 0.3,
                "top_p": 0.95,
            },
        )

    def process_conversation(self, user_query: str, conversation_id: str | None = None) -> dict:
        """
        Process a chatbot conversation request.

        Args:
            user_query: The user's message
            conversation_id: Optional conversation ID to continue an existing conversation

        Returns:
            dict with 'content' (response text) and 'conversation_id' (string or None)
        """
        # Moderate input
        if not self._moderate_input(user_query):
            return {
                "content": "I can not answer this question.",
                "conversation_id": conversation_id,
            }

        # Start or continue conversation
        if conversation_id is None:
            response = self.mistral_client.beta.conversations.start(
                agent_id=self.agent.id, inputs=user_query
            )
        else:
            response = self.mistral_client.beta.conversations.append(
                conversation_id=conversation_id, inputs=user_query
            )

        # Handle function calls
        result = self._handle_function_call(response)

        return {
            "content": result,
            "conversation_id": response.conversation_id,
        }

    def _handle_function_call(self, input_response: ConversationResponse) -> str:
        """Handle function calls recursively."""
        if input_response.outputs[-1].type == "function.call":
            function_args = json.loads(input_response.outputs[-1].arguments)
            function_name = input_response.outputs[-1].name

            if function_name in self.tools.keys():
                function_result = json.dumps(self.tools[function_name](**function_args))
            else:
                return input_response.outputs[-1].content

            # Providing the result to our Agent
            user_function_calling_entry = FunctionResultEntry(
                tool_call_id=input_response.outputs[-1].tool_call_id,
                result=function_result,
            )
            # Retrieving the final response
            tool_response = self.mistral_client.beta.conversations.append(
                conversation_id=input_response.conversation_id,
                inputs=[user_function_calling_entry],
            )
            return self._handle_function_call(tool_response)
        else:
            return input_response.outputs[-1].content

    def _moderate_input(self, input_query: str) -> bool:
        """Moderate user input for inappropriate content."""
        response = self.mistral_client.classifiers.moderate_chat(
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

    def get_past_conversation_messages(self, conversation_id: str) -> ConversationMessages:
        """Get conversation history."""
        return self.mistral_client.beta.conversations.get_messages(conversation_id=conversation_id)
