from typing import Any

from pydantic import BaseModel, field_validator


class BaseMetadataTemplate(BaseModel):
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
