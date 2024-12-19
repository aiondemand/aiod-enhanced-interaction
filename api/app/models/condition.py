from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator


class Expression(BaseModel):
    value: str | int
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

    # TODO ugly typechecking -> we need to check it with dynamic types that are built
    # for the second stage of LLM user query processing I suppose...
    # def correct_expression_values(self) -> Condition:
    #     data_type = (
    #         int
    #         if self.field
    #         in ["size_in_mb", "datapoints_lower_bound", "datapoints_upper_bound"]
    #         else str
    #     )
    #     for expr in self.expressions:
    #         try:
    #             expr.value = data_type(expr.value)
    #         except:
    #             ValueError("Invalid expression value")

    #     return self

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
