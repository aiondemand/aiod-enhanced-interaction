from copy import deepcopy
from functools import partial
from types import UnionType
from typing import Annotated, Any, Callable, Literal, Type, Union, get_args, get_origin

from app.schemas.asset_metadata.base import BaseMetadataTemplate
from app.schemas.asset_metadata.dataset_metadata import (
    HuggingFaceDatasetMetadataTemplate,
)
from app.schemas.enums import AssetType
from pydantic import BaseModel, field_validator
from pydantic._internal._decorators import Decorator, FieldValidatorDecoratorInfo
from pydantic.fields import FieldInfo


class SchemaOperations:
    SCHEMA_MAPPING: dict[AssetType, Type[BaseMetadataTemplate]] = {
        AssetType.DATASETS: HuggingFaceDatasetMetadataTemplate
    }

    @classmethod
    def get_asset_schema(cls, asset_type: AssetType) -> Type[BaseMetadataTemplate]:
        return cls.SCHEMA_MAPPING[asset_type]

    @classmethod
    def get_supported_asset_types(cls) -> list[AssetType]:
        return list(cls.SCHEMA_MAPPING.keys())

    @classmethod
    def get_schema_field_names(cls, asset_schema: Type[BaseModel]) -> list[str]:
        return list(asset_schema.model_fields.keys())

    @classmethod
    def strip_outer_annotation(
        cls, asset_schema: Type[BaseMetadataTemplate], field_name: str
    ) -> Type:
        outer_annot = asset_schema.model_fields[field_name].annotation
        if outer_annot is None:
            raise ValueError(f"Field {field_name} does not have an annotation")
        return cls.strip_list_type(cls.strip_optional_type(outer_annot))

    @classmethod
    def get_inner_annotation(
        cls, asset_schema: Type[BaseMetadataTemplate], field_name: str
    ) -> Type:
        annotated_type = cls.strip_outer_annotation(asset_schema, field_name)
        if not cls.is_valid_inner_annotation(annotated_type):
            raise ValueError("Invalid metadata filtering schema")

        inner_annotation = get_args(annotated_type)[0]
        if inner_annotation is None:
            raise ValueError(f"Field {field_name} does not have an inner annotation")
        return inner_annotation

    @classmethod
    def get_inner_field_info(
        cls, asset_schema: Type[BaseMetadataTemplate], field_name: str
    ) -> FieldInfo:
        annotated_type = cls.strip_outer_annotation(asset_schema, field_name)
        if not cls.is_valid_inner_annotation(annotated_type):
            raise ValueError("Invalid metadata filtering schema")

        inner_annotation = cls.get_inner_annotation(asset_schema, field_name)
        field_info: FieldInfo = deepcopy(get_args(annotated_type)[1])
        field_info.annotation = inner_annotation

        return field_info

    @classmethod
    def exists_list_of_valid_values(
        cls, asset_schema: Type[BaseMetadataTemplate], field_name: str
    ) -> bool:
        inner_annotations_class = asset_schema.get_inner_annotations_class()
        return inner_annotations_class.exists_list_of_valid_values(field_name)

    @classmethod
    def get_list_of_valid_values(
        cls, asset_schema: Type[BaseMetadataTemplate], field_name: str
    ) -> list[str]:
        inner_annotations_class = asset_schema.get_inner_annotations_class()
        if not inner_annotations_class.exists_list_of_valid_values(field_name):
            raise ValueError("Field does not have a list of valid values")

        return inner_annotations_class.get_list_of_valid_values(field_name)

    @classmethod
    def is_valid_inner_annotation(cls, annotated_type: Type) -> bool:
        return (
            get_origin(annotated_type) is Annotated
            and len(get_args(annotated_type)) == 2
            and isinstance(get_args(annotated_type)[1], FieldInfo)
        )

    @classmethod
    def get_list_fields_mask(cls, asset_schema: Type[BaseModel]) -> dict[str, bool]:
        return {
            k: cls.is_list_type(cls.strip_optional_type(v))
            for k, v in asset_schema.__annotations__.items()
        }

    @classmethod
    def is_optional_type(cls, annotation: Type) -> bool:
        if get_origin(annotation) is Union or get_origin(annotation) is UnionType:
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
    def get_inner_most_primitive_type(cls, annotation: Type | None) -> Type:
        if annotation is None:
            raise ValueError("Annotation is None")

        origin = get_origin(annotation)
        if origin is Literal:
            return type(get_args(annotation)[0])
        if origin is not None:
            args = get_args(annotation)
            if args:
                # Check the first argument for simplicity sake
                return cls.get_inner_most_primitive_type(args[0])
        return annotation

    @classmethod
    def translate_primitive_type_to_str(cls, annotation: Type) -> str:
        name_mapping = {str: "string", int: "integer", float: "float", bool: "boolean"}

        if annotation not in name_mapping:
            raise ValueError("Not supported data type")
        return name_mapping[annotation]

    @classmethod
    def _orig_field_validator_wrapper(
        cls, value: Any, func: Callable, asset_schema: Type[BaseMetadataTemplate], field_name: str
    ) -> Any:
        if value is None:
            return None

        is_list_field = cls.get_list_fields_mask(asset_schema)[field_name]
        if is_list_field is False:
            return func(value)
        if is_list_field:
            out = func([value])
            if len(out) > 0:
                return out[0]
        raise ValueError(
            f"Value '{str(value)}' didn't comply with '{field_name}' validator demands"
        )

    @classmethod
    def create_new_field_validators(
        cls, asset_schema: Type[BaseMetadataTemplate], orig_field_name: str, new_field_name: str
    ) -> dict:
        validators = SchemaOperations.get_field_validators(asset_schema, orig_field_name)

        validators_dict = {
            # wrapping original field validators in '_orig_field_validator_wrapper' function
            f"{func_name}_wrapper": field_validator(new_field_name, mode=decor.info.mode)(
                partial(
                    cls._orig_field_validator_wrapper,
                    func=getattr(asset_schema, func_name),
                    asset_schema=asset_schema,
                    field_name=orig_field_name,
                )
            )
            for func_name, decor in validators
        }
        return validators_dict
