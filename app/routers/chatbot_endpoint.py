from fastapi import APIRouter, HTTPException, Query
from mistralai import SDKError
from app.schemas.chatbot import ChatbotHistory, ChatbotResponse
from app.services.chatbot.chatbot import (
    start_conversation,
    continue_conversation,
    get_past_conversation_messages,
)

router = APIRouter()


# Github issue: https://github.com/aiondemand/aiod-enhanced-interaction/issues/126
# TODO stream the chatbot responses to make it more interactive


@router.post("")
async def answer_query(
    user_query: str = Query(..., description="User query"),
    conversation_id: str | None = Query(
        default=None, description="Conversation ID. Leave empty to start a new conversation."
    ),
) -> ChatbotResponse:
    """
    Handles user queries, either starting a new conversation or continuing an existing one
    based on the presence of a conversation ID.
    When this router is included with a prefix like /chatbot, this endpoint becomes /chatbot.
    """
    try:
        if conversation_id is None:
            response_content, conversation_id = await start_conversation(user_query)
        else:
            response_content = await continue_conversation(user_query, conversation_id)
    except SDKError as e:
        if e.status_code == 429:
            raise HTTPException(
                status_code=503,
                detail="You have exceeded the chatbot limits. Try the request again in a few minutes.",
            )
        else:
            raise e

    return ChatbotResponse(
        conversation_id=conversation_id,
        content=response_content,
    )


@router.get("/history")
async def get_history(
    conversation_id: str | None = Query(
        default=None, description="Conversation ID returned from the chatbot endpoint."
    ),
) -> ChatbotHistory | None:
    """
    Returns the conversation history for the current user.
    """
    if conversation_id is None:
        return None

    history = await get_past_conversation_messages(conversation_id)
    return ChatbotHistory.create_from_mistral_history(history)
