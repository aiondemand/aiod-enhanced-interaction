from __future__ import annotations

from datetime import datetime
from typing import Literal
from mistralai import ConversationMessages, MessageEntries
from pydantic import BaseModel


class ChatbotResponse(BaseModel):
    conversation_id: str | None = None
    content: str


# TODO this model can only represent communication between a user and a model
# whilst ignoring various tool calls performed on the side of the agent
# Since currently we currently extract from Mistral API only the messages
# that are directly a part of user interaction, it's not a problem for now...
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
