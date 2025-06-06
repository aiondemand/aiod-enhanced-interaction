from typing import Union
from fastapi import FastAPI, Cookie
from app.chatbot.chatbot_main_mistral import start_conversation
from pydantic import BaseModel

app = FastAPI()  # probably change to APIRouter in the long run, keep fastapi for testing


class Cookies(BaseModel):
    conversation_id: str | None = None


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/talk2aiod/{query}")
def answer_query(query: str):
    result, conversation_id = start_conversation(query)

    return result
