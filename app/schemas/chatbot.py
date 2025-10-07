from __future__ import annotations

from datetime import datetime
from typing import Literal
from mistralai import ConversationMessages, MessageEntries
from pydantic import BaseModel


class ChatbotResponse(BaseModel):
    conversation_id: str | None = None
    content: str


class ChatbotMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime

    @staticmethod
    def create_from_mistral_message(message: MessageEntries) -> ChatbotMessage:
        return ChatbotMessage(
            role=message.role,
            content=message.content,
            created_at=message.created_at,
        )


class ChatbotHistory(BaseModel):
    conversation_id: str
    messages: list[ChatbotMessage]

    @staticmethod
    def create_from_mistral_history(history: ConversationMessages) -> ChatbotHistory:
        return ChatbotHistory(
            conversation_id=history.conversation_id,
            messages=[ChatbotMessage.create_from_mistral_message(mes) for mes in history.messages],
        )
