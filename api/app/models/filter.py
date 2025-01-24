from __future__ import annotations

from typing import Literal

from app.schemas.asset_metadata.base import SchemaOperations
from app.schemas.enums import AssetType
from fastapi import HTTPException
from pydantic import BaseModel


class Expression(BaseModel):
    value: str | int | float
    comparison_operator: Literal["<", ">", "<=", ">=", "==", "!="]


class Filter(BaseModel):
    field: str
    logical_operator: Literal["AND", "OR"]
    expressions: list[Expression]

    def validate_filter_or_raise(self, asset_type: AssetType) -> None:
        asset_schema = SchemaOperations.get_asset_schema(asset_type)
        if self.field not in SchemaOperations.get_schema_field_names(asset_schema):
            raise HTTPException(
                status_code=400, detail=f"Invalid field value '{self.field}'"
            )

        for expr in self.expressions:
            if (
                SchemaOperations.validate_value_against_type(
                    expr.value, asset_schema, self.field
                )
                is False
            ):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid value '{str(expr.value)}' for field '{self.field}'",
                )

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
                    Expression(value="10000", comparison_operator=">"),
                ],
            ).model_dump(),
        ]
