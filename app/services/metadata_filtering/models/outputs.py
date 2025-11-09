from typing import Literal, TypeAlias
from pydantic import BaseModel, Field


PrimitiveTypes: TypeAlias = int | float | str | bool


class LLMExpression(BaseModel):
    """An Expression represents a single comparison between a value and a metadata field"""

    raw_value: str = Field(
        ..., description="Raw value directly extracted from the natural language condition"
    )
    processed_value: PrimitiveTypes | None = Field(
        ...,
        description=(
            "Transformed original value adhering to value constraints of the specific metadata field. "
            "Value to be compared to the metadata field. "
            "If the original value cannot be unambiguosly mapped to one of the valid values, this field is set to None."
        ),
    )
    comparison_operator: Literal["<", ">", "<=", ">=", "==", "!="] = Field(
        ...,
        description="Comparison operator to be used for comparing the value to the metadata field",
    )
    discard: bool = Field(
        ...,
        description="If the value found within the natural language condition cannot be unambiguously mapped to its valid counterpart value, set this to True",
    )


class LLMStructedCondition(BaseModel):
    """A Condition consists of one or more expressions joined with a logical operator"""

    field: str = Field(..., description="Name of the metadata field to filter by")
    logical_operator: Literal["AND", "OR"] = Field(
        ..., description="Allowed logical operator to be used for combining multiple expressions"
    )
    expressions: list[LLMExpression] = Field(
        ...,
        description="List of expressions associated with their respective values and comparison operators to be used for filtering",
    )


class LLM_NaturalLanguageCondition(BaseModel):
    """Condition in its natural language form extracted from user query"""

    condition: str = Field(
        ...,
        description=(
            "Natural language condition corresponding to a particular metadata field we use for filtering. "
            "It may contain either only one value to be compared to metadata field, "
            "or multiple values if there's an OR logical operator in between those values"
        ),
    )
    field: str = Field(..., description="Name of the metadata field")
    operator: Literal["AND", "OR", "NONE"] = Field(
        ...,
        description=(
            "Logical operator used between multiple values pertaining to the same metadata field. "
            "If the condition describes only one value, set it to NONE instead."
        ),
    )


class LLM_NormalizedValue(BaseModel):
    original_value: str = Field(..., description="The input value before normalization.")
    normalized_value: str | None = Field(
        ..., description="The resulting value after applying normalization rules."
    )
