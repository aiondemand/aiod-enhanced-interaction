from abc import ABC, abstractmethod
from typing import Any, Generic, Type, TypeVar

from pydantic import BaseModel, field_validator


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
    @field_validator("*", mode="before")
    @classmethod
    def convert_strings_to_lowercase(cls, value: Any) -> Any:
        return cls.apply_lowercase_recursively(value)

    @classmethod
    def apply_lowercase_recursively(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.lower()
        elif isinstance(value, list):
            return [cls.apply_lowercase_recursively(item) for item in value]
        elif isinstance(value, dict):
            return {k: cls.apply_lowercase_recursively(v) for k, v in value.items()}
        elif isinstance(value, tuple):
            return tuple(cls.apply_lowercase_recursively(item) for item in value)
        elif isinstance(value, set):
            return {cls.apply_lowercase_recursively(item) for item in value}
        else:
            return value

    @classmethod
    @abstractmethod
    def get_inner_annotations_class(cls) -> Type[InnerAnnotations]:
        raise NotImplementedError
