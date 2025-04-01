from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Type, TypeVar

from pydantic import BaseModel


class BaseInnerAnnotations(ABC):
    @classmethod
    @abstractmethod
    def exists_list_of_valid_values(cls, field_name: str) -> bool:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_list_of_valid_values(cls, field_name: str) -> list[str]:
        raise NotImplementedError


InnerAnnotations = TypeVar("InnerAnnotations", bound=BaseInnerAnnotations)


class BaseMetadataTemplate(BaseModel, Generic[InnerAnnotations], ABC):
    @classmethod
    def apply_string_function_recursively(
        cls, value: Any, string_function: Callable[[str], str]
    ) -> Any:
        if isinstance(value, str):
            return string_function(value)
        elif isinstance(value, list):
            return [cls.apply_string_function_recursively(item, string_function) for item in value]
        elif isinstance(value, dict):
            return {
                k: cls.apply_string_function_recursively(v, string_function)
                for k, v in value.items()
            }
        elif isinstance(value, tuple):
            return tuple(
                cls.apply_string_function_recursively(item, string_function) for item in value
            )
        elif isinstance(value, set):
            return {cls.apply_string_function_recursively(item, string_function) for item in value}
        else:
            return value

    @classmethod
    @abstractmethod
    def get_inner_annotations_class(cls) -> Type[InnerAnnotations]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_supported_comparison_operators(cls, field_name: str) -> list[str]:
        raise NotImplementedError
