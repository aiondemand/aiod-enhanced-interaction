import logging
import time
from functools import partial, wraps
from typing import Any, Callable, Type, TypeVar

from app.config import settings


class ServiceUnavailableException(Exception):
    """Base exception class for when a service is unavailable"""

    pass


class LocalServiceUnavailableException(ServiceUnavailableException):
    """Exception for when a local service (running on the same VM) is unavailable"""

    pass


class MilvusUnavailableException(LocalServiceUnavailableException):
    """Exception for when the local Milvus vector database service is unavailable"""

    pass


class OllamaUnavailableException(LocalServiceUnavailableException):
    """Exception for when the local Ollama LLM service is unavailable"""

    pass


class AIoDUnavailableException(ServiceUnavailableException):
    """Exception for when the external AIoD service is unavailable"""

    pass


T = TypeVar("T")


def with_retry_sync(
    exception_cls: Type[ServiceUnavailableException] = ServiceUnavailableException,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            max_retries = settings.RETRY_RETRIES
            sleep_time = settings.RETRY_SLEEP_TIME

            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e
                    logging.warning(
                        f"Function '{func}' failed (attempt {attempt + 1}/{max_retries}): {str(e)}"
                    )
                    if attempt < max_retries - 1:
                        time.sleep(sleep_time)

            raise exception_cls(
                f"{exception_cls.__name__}: Service appears to be down or unresponsive after {max_retries} attempts. Last error: {str(last_exception)}"
            )

        return wrapper

    return decorator


# TODO incorporate it into Ollama and LLM invocations
class OllamaClientResilientWrapper:
    def __init__(self, uri: str | None = None) -> None:
        self.uri = uri
        self.timeout = settings.OLLAMA.TIMEOUT

    def __getattribute__(self, name: str) -> Any:
        try:
            func = super().__getattribute__(name)
        except AttributeError:
            return None

        if not callable(func) or name.startswith("__"):
            return func

        # TODO there may be no timeout argument in the function
        return with_retry_sync(exception_cls=OllamaUnavailableException)(
            partial(func, timeout=self.timeout)
        )
