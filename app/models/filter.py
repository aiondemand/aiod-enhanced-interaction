from __future__ import annotations

from typing import Literal, Type

from fastapi import HTTPException
from pydantic import BaseModel, Field, ValidationError

from app.schemas.asset_metadata.types import ComparisonOperator, LogicalOperator, PrimitiveTypes
from app.schemas.enums import SupportedAssetType
from app.services.metadata_filtering.models.outputs import LLMStructedCondition
from app.services.metadata_filtering.schema_mapping import QUERY_PARSING_SCHEMA_MAPPING


class Expression(BaseModel):
    """An Expression represents a single comparison between a value and a metadata field"""

    value: PrimitiveTypes = Field(..., description="Value to be compared to the metadata field")
    comparison_operator: ComparisonOperator = Field(
        ...,
        description="Comparison operator to be used for comparing the value to the metadata field",
    )


class Filter(BaseModel):
    """A filter consists of one or more expressions joined with a logical operator"""

    field: str = Field(..., max_length=30, description="Name of the metadata field to filter by")
    logical_operator: LogicalOperator = Field(
        ..., description="Allowed logical operator to be used for combining multiple expressions"
    )
    expressions: list[Expression] = Field(
        ...,
        description="List of expressions associated with their respective values and comparison operators to be used for filtering",
        max_length=5,
    )

    @staticmethod
    def build_from_llm_condition(condition: LLMStructedCondition) -> Filter:
        kwargs = condition.model_dump()
        for it in range(len(kwargs["expressions"])):
            kwargs["expressions"][it]["value"] = kwargs["expressions"][it].pop("processed_value")

        return Filter(**kwargs)

    @classmethod
    def create_field_specific_filter_type(
        cls, asset_type: SupportedAssetType, field_name: str
    ) -> Type[BaseModel]:
        asset_schema = QUERY_PARSING_SCHEMA_MAPPING[asset_type]

        if field_name not in list(asset_schema.model_fields.keys()):
            raise HTTPException(status_code=400, detail=f"Invalid field value '{field_name}'")

        expression_class = cls._create_field_specific_expression_type(asset_type, field_name)

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
                ...,
                description=Filter.model_fields["expressions"].description,
                max_length=5,
            ),
        }
        return type(
            f"Filter_{asset_type.value.capitalize()}_{field_name.capitalize()}",
            (BaseModel,),
            filter_class_dict,
        )

    @classmethod
    def _create_field_specific_expression_type(
        cls, asset_type: SupportedAssetType, field_name: str
    ) -> Type[BaseModel]:
        asset_schema = QUERY_PARSING_SCHEMA_MAPPING[asset_type]
        annotation = asset_schema.get_inner_annotation(field_name, with_valid_values_enum=True)

        expression_class_dict = {
            "__annotations__": {
                "value": annotation,
                "comparison_operator": Literal[
                    *asset_schema.get_supported_comparison_operators(field_name)
                ],
            },
            "value": Field(..., description=asset_schema.model_fields[field_name].description),
            "comparison_operator": Field(
                ..., description=Expression.model_fields["comparison_operator"].description
            ),
        }
        return type(
            f"Expression_{asset_type.value.capitalize()}_{field_name.capitalize()}",
            (BaseModel,),
            expression_class_dict,
        )

    def validate_filter_or_raise(self, asset_type: SupportedAssetType) -> None:
        filter_class = self.create_field_specific_filter_type(asset_type, self.field)

        try:
            validated_filter = filter_class(**self.model_dump())
            validated_filter = Filter(**validated_filter.model_dump())

            # update the values within expressions in the case of implicit data type conversion
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
                field="modalities",
                logical_operator="AND",
                expressions=[
                    Expression(value="image", comparison_operator="=="),
                    Expression(value="text", comparison_operator="!="),
                ],
            ).model_dump(),
        ]
