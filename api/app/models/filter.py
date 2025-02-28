from __future__ import annotations

from typing import Literal, TypeAlias

from fastapi import HTTPException
from pydantic import BaseModel, Field, field_validator

from app.schemas.asset_metadata.base import SchemaOperations
from app.schemas.enums import AssetType

PrimitiveTypes: TypeAlias = str | int | float


class Expression(BaseModel):
    value: PrimitiveTypes
    comparison_operator: Literal["<", ">", "<=", ">=", "==", "!="]

    @field_validator("value", mode="before")
    @classmethod
    def constrain_length(cls, value: PrimitiveTypes) -> PrimitiveTypes:
        if isinstance(value, str) and len(value) > 50:
            raise ValueError("Value length must be less than 50 characters")

        return value


class Filter(BaseModel):
    field: str = Field(..., max_length=30)
    logical_operator: Literal["AND", "OR"]
    expressions: list[Expression]

    def validate_filter_or_raise(self, asset_type: AssetType) -> None:
        asset_schema = SchemaOperations.get_asset_schema(asset_type)
        if self.field not in SchemaOperations.get_schema_field_names(asset_schema):
            raise HTTPException(status_code=400, detail=f"Invalid field value '{self.field}'")

        for expr in self.expressions:
            validated_value = SchemaOperations.validate_value_against_type(
                expr.value, asset_schema, self.field
            )
            if validated_value is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid value '{str(expr.value)}' for field '{self.field}'",
                )
            expr.value = validated_value

    @classmethod
    def get_body_examples(cls) -> list[dict]:
        return [
            Filter(
                field="languages",
                logical_operator="AND",
                expressions=[
                    Expression(value="en", comparison_operator="=="),
                    Expression(value="es", comparison_operator="=="),
                ],
            ).model_dump(),
            Filter(
                field="datapoints_lower_bound",
                logical_operator="AND",
                expressions=[
                    Expression(value=10000, comparison_operator=">"),
                ],
            ).model_dump(),
        ]
