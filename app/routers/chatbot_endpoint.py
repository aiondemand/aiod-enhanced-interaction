from fastapi import APIRouter, Response, Request, Query

from app.services.chatbot.chatbot import start_conversation, continue_conversation

router = APIRouter()


CONVERSATION_ID_COOKIE_KEY = "chat_continue_conversation"
COOKIE_PATH = "/chatbot"

# Github issue: https://github.com/aiondemand/aiod-enhanced-interaction/issues/126
# TODO stream the chatbot responses to make it more interactive


@router.post("")
async def answer_query(
    request: Request, user_query: str = Query(..., description="User query")
) -> Response:
    """
    Handles user queries, either starting a new conversation or continuing an existing one
    based on the presence of a conversation ID cookie.
    When this router is included with a prefix like /chatbot, this endpoint becomes /chatbot.
    """
    conversation_id: str | None = request.cookies.get(CONVERSATION_ID_COOKIE_KEY)

    if not conversation_id:
        # If no conversation ID cookie is found, start a new conversation
        response_content, new_conversation_id = start_conversation(user_query)

        # Create the Response object here
        final_response = Response(content=response_content, media_type="text/plain")

        # Check whether answering to the user query was allowed
        if new_conversation_id != "-1":
            # Set the new conversation ID as a cookie directly on the final_response object
            # Set the cookie path to match the router prefix
            final_response.set_cookie(
                key=CONVERSATION_ID_COOKIE_KEY,
                value=new_conversation_id,
                httponly=True,
                samesite="lax",
                path=COOKIE_PATH,  # Using the explicit cookie_path
                secure=True,  # Set to True if deploying with HTTPS
            )
    else:
        # If a conversation ID cookie exists, continue the existing conversation
        response_content = continue_conversation(user_query, conversation_id)
        # For subsequent requests, if you need to return a Response object
        # but don't need to set a *new* cookie, you can create it here:
        final_response = Response(content=response_content, media_type="text/plain")

    return final_response  # Return the Response object on which the cookie was set


@router.post("/clear_conversation")
async def clear_conversation(response: Response) -> None:
    """
    Clears the conversation ID cookie from the client's browser.
    Note: To properly clear a cookie, you need to set its expiration to a past date
    or use delete_cookie().
    """
    response.delete_cookie(
        key=CONVERSATION_ID_COOKIE_KEY,
        httponly=True,
        samesite="lax",
        path=COOKIE_PATH,  # Using the explicit cookie_path
        secure=True,  # Set to True if deploying with HTTPS
    )
