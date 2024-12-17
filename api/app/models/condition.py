# TODO
# for now we shall copy the ones used for LLM output JSON schema
# later on we may utilize them directly instead

from typing import Literal

from pydantic import BaseModel, field_validator


class Expression(BaseModel):
    value: str | int | float
    comparison_operator: Literal["<", ">", "<=", ">=", "==", "!="]


class Condition(BaseModel):
    field: str
    logical_operator: Literal["AND", "OR"]
    expressions: list[Expression]

    @field_validator("field", mode="before")
    @classmethod
    def validate_field(cls, value: str) -> str:
        assert value in cls.get_valid_fields(), "Invalid field for metadata filtering"
        return value

    @classmethod
    def get_valid_fields(cls) -> list[str]:
        # TODO
        # Hardcoded list of fields we use for conditions
        return (
            "date_published",
            "size_in_mb",
            "license",
            "task_types",
            "languages",
            "datapoints_lower_bound",
            "datapoints_upper_bound",
        )
