from __future__ import annotations

from typing import Annotated, Literal, Type, TypeAlias

from fastapi import HTTPException
from pydantic import BaseModel, Field, ValidationError, field_validator

from app.schemas.asset_metadata.operations import SchemaOperations
from app.schemas.enums import AssetType

PrimitiveTypes: TypeAlias = int | float | Annotated[str, Field(max_length=50)]


class Expression(BaseModel):
    value: PrimitiveTypes = Field(..., description="Value to be compared to the metadata field")
    comparison_operator: Literal["<", ">", "<=", ">=", "==", "!="] = Field(
        ...,
        description="Eligible comparison operator to be used for comparing the value to the metadata field",
    )


class Filter(BaseModel):
    field: str = Field(..., max_length=30, description="Name of the metadata field to filter by")
    logical_operator: Literal["AND", "OR"] = Field(
        ..., description="Eligible logical operator to be used for combining multiple expressions"
    )
    expressions: list[Expression] = Field(
        ...,
        description="List of expressions associated with their respective values and comparison operators to be used for filtering",
        max_length=5,
    )

    @field_validator("field", mode="before")
    @classmethod
    def validate_field(cls, v: str) -> str:
        return v.lower()

    @field_validator("logical_operator", mode="before")
    @classmethod
    def validate_logical_operator(cls, v: str) -> str:
        return v.upper()

    @classmethod
    def create_field_specific_filter_type(
        cls, asset_type: AssetType, field_name: str
    ) -> Type[BaseModel]:
        asset_schema = SchemaOperations.get_asset_schema(asset_type)
        if field_name not in SchemaOperations.get_schema_field_names(asset_schema):
            raise HTTPException(status_code=400, detail=f"Invalid field value '{field_name}'")

        expression_class = cls._create_expression_type(asset_type, field_name)

        filter_class_dict = {
            "__annotations__": {
                "field": Literal[field_name],
                "logical_operator": Filter.model_fields["logical_operator"].annotation,
                "expressions": list[expression_class],  # type: ignore[valid-type]
            },
            "field": Field(..., description=Filter.model_fields["field"].description),
            "logical_operator": Field(
                ..., description=Filter.model_fields["logical_operator"].description
            ),
            "expressions": Field(
                ..., description=Filter.model_fields["expressions"].description, max_length=5
            ),
        }

        field_validators = SchemaOperations.get_field_validators(Filter, "field")
        logical_operator_validators = SchemaOperations.get_field_validators(
            Filter, "logical_operator"
        )
        field_names = ["field"] * len(field_validators) + ["logical_operator"] * len(
            logical_operator_validators
        )

        filter_class_dict.update(
            {
                func_name: field_validator(field_name, mode=decor.info.mode)(
                    getattr(Filter, func_name)
                )
                for (func_name, decor), field_name in zip(
                    field_validators + logical_operator_validators, field_names
                )
            }
        )

        return type(
            f"Filter_{asset_type.value.capitalize()}_{field_name.capitalize()}",
            (BaseModel,),
            filter_class_dict,
        )

    @classmethod
    def _create_expression_type(cls, asset_type: AssetType, field_name: str) -> Type[BaseModel]:
        asset_schema = SchemaOperations.get_asset_schema(asset_type)
        annotation = SchemaOperations.get_inner_annotation(asset_schema, field_name)
        field_info = SchemaOperations.get_inner_field_info(asset_schema, field_name)

        expression_class_dict = {
            "__annotations__": {
                "value": annotation,
                "comparison_operator": Literal[
                    *asset_schema.get_supported_comparison_operators(field_name)
                ],
            },
            "value": field_info,
            "comparison_operator": Field(
                ..., description=Expression.model_fields["comparison_operator"].description
            ),
        }
        expression_class_dict.update(
            SchemaOperations.create_new_field_validators(
                asset_schema, orig_field_name=field_name, new_field_name="value"
            )
        )
        return type(
            f"Expression_{asset_type.value.capitalize()}_{field_name.capitalize()}",
            (BaseModel,),
            expression_class_dict,
        )

    def validate_filter_or_raise(self, asset_type: AssetType) -> None:
        filter_class = self.create_field_specific_filter_type(asset_type, self.field)

        try:
            validated_filter = filter_class(**self.model_dump())
            validated_filter = Filter(**validated_filter.model_dump())

            self.expressions = [
                Expression(**expr.model_dump()) for expr in validated_filter.expressions
            ]
        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))

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
