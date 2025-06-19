from typing import Optional
from fastapi import FastAPI, Cookie, APIRouter, Response, Request
from app.chatbot.chatbot_main_mistral import start_conversation, continue_conversation
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter()


@router.get("/")
def read_root():
    return {"Hello": "Start chatting with Talk2AIoD"}


CONVERSATION_ID_COOKIE_KEY = "chat_continue_conversation"
cookie_path = "/chat"


class QueryRequest(BaseModel):
    query: str


@router.post("/")
async def answer_query(request_body: QueryRequest, request: Request):
    """
    Handles user queries, either starting a new conversation or continuing an existing one
    based on the presence of a conversation ID cookie.
    The query is now passed in the request body.
    When this router is included with a prefix like /chat, this endpoint becomes /chat.
    """
    query = request_body.query
    conversation_id: Optional[str] = request.cookies.get(CONVERSATION_ID_COOKIE_KEY)
    response_content: str

    if not conversation_id:
        # If no conversation ID cookie is found, start a new conversation
        response_content, new_conversation_id = start_conversation(query)

        # Create the Response object here
        final_response = Response(content=response_content, media_type='text/plain')

        # Set the new conversation ID as a cookie directly on the final_response object
        # Set the cookie path to match the router prefix
        final_response.set_cookie(
            key=CONVERSATION_ID_COOKIE_KEY,
            value=new_conversation_id,
            httponly=True,
            samesite="lax",
            path=cookie_path, # Using the explicit cookie_path
            secure=False # Set to True if deploying with HTTPS
        )
    else:
        # If a conversation ID cookie exists, continue the existing conversation
        response_content = continue_conversation(query, conversation_id)
        # For subsequent requests, if you need to return a Response object
        # but don't need to set a *new* cookie, you can create it here:
        final_response = Response(content=response_content, media_type='text/plain')

    return final_response # Return the Response object on which the cookie was set


@router.get("/clear_conversation")  # Changed to POST as clearing is an action
async def clear_conversation(response: Response):
    """
    Clears the conversation ID cookie from the client's browser.
    Note: To properly clear a cookie, you need to set its expiration to a past date
    or use delete_cookie().
    """
    response.delete_cookie(
        key=CONVERSATION_ID_COOKIE_KEY,
        httponly=True,
        samesite="lax",
        path=cookie_path,  # Using the explicit cookie_path
        secure=False  # Set to True if deploying with HTTPS
    )
    return "Cookie removed"



