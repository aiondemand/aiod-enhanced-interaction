import asyncio
import logging
from typing import cast
from fastapi import APIRouter, HTTPException, Query
from mistralai import ConversationMessages, SDKError
from celery.result import AsyncResult

from app.schemas.chatbot import ChatbotHistory, ChatbotResponse
from app.celery_tasks import chatbot_conversation_task, chatbot_history_task

router = APIRouter()

# Github issue: https://github.com/aiondemand/aiod-enhanced-interaction/issues/126
# TODO stream the chatbot responses to make it more interactive

CHATBOT_TASK_TIMEOUT = 60
POLL_INTERVAL = 1


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

    This endpoint dispatches the work to a Celery search worker and polls for results.
    """
    try:
        # Dispatch the task to Celery
        task = chatbot_conversation_task.delay(user_query, conversation_id)
        task_result = await poll_task_result(task)

        return ChatbotResponse(**task_result)
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error processing chatbot query: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request.",
        )


@router.get("/history")
async def get_history(
    conversation_id: str | None = Query(
        default=None, description="Conversation ID returned from the chatbot endpoint."
    ),
) -> ChatbotHistory | None:
    """
    Returns the conversation history for the current user.

    This endpoint dispatches the work to a Celery search worker and polls for results.
    """
    if conversation_id is None:
        return None

    try:
        # Dispatch the task to Celery
        task = chatbot_history_task.delay(conversation_id)
        history = await poll_task_result(task)

        return ChatbotHistory.create_from_mistral_history(ConversationMessages(**history))
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error retrieving conversation history: {e}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while retrieving conversation history.",
        )


async def poll_task_result(task: AsyncResult) -> dict:
    elapsed_time = 0

    while elapsed_time < CHATBOT_TASK_TIMEOUT:
        if task.ready():
            if task.successful():
                return cast(dict, task.get())
            else:
                # Task failed, get the exception
                try:
                    task.get()
                except SDKError as e:
                    if e.status_code == 429:
                        raise HTTPException(
                            status_code=503,
                            detail="You have exceeded the chatbot limits. Try the request again in a few minutes.",
                        )
                    else:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Chatbot error: {str(e)}",
                        )
                except Exception as e:
                    logging.error(f"Chatbot task failed: {e}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Chatbot task failed: {str(e)}",
                    )

        await asyncio.sleep(POLL_INTERVAL)
        elapsed_time += POLL_INTERVAL

    # Timeout reached
    raise HTTPException(
        status_code=504,
        detail="Chatbot request timed out. Please try again.",
    )
