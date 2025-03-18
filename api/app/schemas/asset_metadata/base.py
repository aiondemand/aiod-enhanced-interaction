from functools import partial
from typing import Any, Callable, Type, Union, get_args, get_origin

from app.schemas.asset_metadata.dataset_metadata import (
    HuggingFaceDatasetMetadataTemplate,
)
from app.schemas.enums import AssetType
from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic._internal._decorators import Decorator, FieldValidatorDecoratorInfo


class SchemaOperations:
    SCHEMA_MAPPING = {AssetType.DATASETS: HuggingFaceDatasetMetadataTemplate}

    @classmethod
    def get_asset_schema(cls, asset_type: AssetType) -> Type[BaseModel]:
        return cls.SCHEMA_MAPPING[asset_type]

    @classmethod
    def get_schema_field_names(cls, asset_schema: Type[BaseModel]) -> list[str]:
        return list(asset_schema.model_fields.keys())

    @classmethod
    def dynamically_create_type_for_a_field_value(
        cls, asset_schema: Type[BaseModel], field_name: str
    ) -> Type:
        original_field = asset_schema.model_fields[field_name]
        if original_field.annotation is None:
            raise ValueError(f"Field '{field_name}' has no annotation")

        return cls.strip_list_type(cls.strip_optional_type(original_field.annotation))

    @classmethod
    def get_list_fields_mask(cls, asset_schema: Type[BaseModel]) -> dict[str, bool]:
        return {
            k: cls.is_list_type(cls.strip_optional_type(v))
            for k, v in asset_schema.__annotations__.items()
        }

    @classmethod
    def is_optional_type(cls, annotation: Type) -> bool:
        if get_origin(annotation) is Union:
            return type(None) in get_args(annotation)
        return False

    @classmethod
    def is_list_type(cls, annotation: Type) -> bool:
        return get_origin(annotation) is list

    @classmethod
    def strip_optional_type(cls, annotation: Type) -> Type:
        if cls.is_optional_type(annotation):
            return next(arg for arg in get_args(annotation) if arg is not type(None))
        return annotation

    @classmethod
    def strip_list_type(cls, annotation: Type) -> Type:
        if cls.is_list_type(annotation):
            return get_args(annotation)[0]
        return annotation

    @classmethod
    def get_field_validators(
        cls, asset_schema: Type[BaseModel], field: str
    ) -> list[tuple[str, Decorator[FieldValidatorDecoratorInfo]]]:
        return [
            (func_name, decor)
            for func_name, decor in asset_schema.__pydantic_decorators__.field_validators.items()
            if field in decor.info.fields
        ]

    @classmethod
    def validate_value_against_type(
        cls, orig_value: Any, asset_schema: Type[BaseModel], field: str
    ) -> Any | None:
        def validate_func(cls, value: Any, func: Callable) -> Any:
            is_list_field = SchemaOperations.get_list_fields_mask(asset_schema)[field]

            if is_list_field is False:
                return func(value)
            if is_list_field:
                out = func([value])
                if len(out) > 0:
                    return out[0]
            raise ValueError(f"Value '{str(value)}' didn't comply with '{field}' validator demands")

        value_type = cls.dynamically_create_type_for_a_field_value(asset_schema, field)
        validators = cls.get_field_validators(asset_schema, field)

        clazz_dict = {"__annotations__": {"value": value_type}, "value": Field(...)}
        clazz_dict.update(
            {
                f"validator_{func_name}": field_validator("value", mode=decor.info.mode)(
                    partial(validate_func, func=getattr(asset_schema, func_name))
                )
                for func_name, decor in validators
            }
        )

        clazz = type(f"Validate_Field_{field}", (BaseModel,), clazz_dict)
        try:
            return clazz(value=orig_value).value
        except ValidationError:
            return None
