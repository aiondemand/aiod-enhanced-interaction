import asyncio
import inspect
import logging
from functools import wraps
from typing import Any, Callable, Type, TypeVar

from tenacity import (
    before_sleep_log,
    retry,
    wait_fixed,
    stop_after_attempt,
    retry_if_exception_type,
)

from app.config import settings


class ServiceUnavailableException(Exception):
    """Base exception class for when a service is unavailable"""

    def __init__(self, last_exception: Exception, service_name: str = "[Service]") -> None:
        super().__init__(
            f"{service_name} is unavailable after having performed {settings.CONNECTION_NUM_RETRIES} attempts: {str(last_exception)}"
        )


class LocalServiceUnavailableException(ServiceUnavailableException):
    """Exception for when a local service (running on the same VM) is unavailable"""

    def __init__(self, last_exception: Exception, service_name: str = "[Local Service]") -> None:
        super().__init__(last_exception, service_name)


class MilvusUnavailableException(LocalServiceUnavailableException):
    """Exception for when the local Milvus vector database service is unavailable"""

    def __init__(self, last_exception: Exception, service_name: str = "[Milvus Database]") -> None:
        super().__init__(last_exception, service_name)


class OllamaUnavailableException(LocalServiceUnavailableException):
    """Exception for when the local Ollama LLM service is unavailable"""

    def __init__(self, last_exception: Exception, service_name: str = "[Ollama Service]") -> None:
        super().__init__(last_exception, service_name)


class AIoDUnavailableException(ServiceUnavailableException):
    """Exception for when the external AIoD service is unavailable"""

    def __init__(
        self, last_exception: Exception, service_name: str = "[AIoD API Catalogue]"
    ) -> None:
        super().__init__(last_exception, service_name)


class MongoUnavailableException(LocalServiceUnavailableException):
    """Exception for when the local MongoDB database service is unavailable"""

    def __init__(self, last_exception: Exception, service_name: str = "[MongoDB Database]") -> None:
        super().__init__(last_exception, service_name)


T = TypeVar("T")


def retry_loop(
    output_exception_cls: Type[ServiceUnavailableException] = ServiceUnavailableException,
) -> Callable[..., Callable]:
    def decorator(func: Callable) -> Callable:
        retry_kwargs = {
            "retry": retry_if_exception_type((Exception,)),
            "wait": wait_fixed(settings.CONNECTION_SLEEP_TIME),
            "stop": stop_after_attempt(settings.CONNECTION_NUM_RETRIES),
            "before_sleep": before_sleep_log(logging.getLogger(__name__), logging.INFO),
            "reraise": True,
        }

        # whether it is an async function
        if asyncio.iscoroutinefunction(inspect.unwrap(func)):

            @retry(**retry_kwargs)
            @wraps(func)
            async def _inner_async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await func(*args, **kwargs)

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return await _inner_async_wrapper(*args, **kwargs)
                except Exception as e:
                    raise output_exception_cls(e)

            return async_wrapper
        else:

            @retry(**retry_kwargs)
            @wraps(func)
            def _inner_sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return _inner_sync_wrapper(*args, **kwargs)
                except Exception as e:
                    raise output_exception_cls(e)

            return sync_wrapper

    return decorator
