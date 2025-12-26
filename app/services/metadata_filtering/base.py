import logging
import os
from urllib.parse import urljoin
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app import settings
from ollama import Client

from app import settings


def prepare_ollama_model() -> OpenAIChatModel:
    try:
        client = Client(host=str(settings.OLLAMA.URI))
        client.pull(settings.OLLAMA.MODEL_NAME)

        ollama_url = urljoin(str(settings.OLLAMA.URI), "v1")
        return OpenAIChatModel(
            model_name=settings.OLLAMA.MODEL_NAME,
            provider=OpenAIProvider(base_url=ollama_url),
        )
    except Exception as e:
        logging.error(e)
        logging.error("Ollama is unavailable. Application is being terminated now")
        os._exit(1)
