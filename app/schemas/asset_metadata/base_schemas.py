from abc import abstractmethod
from typing import List, Optional, cast
from pydantic import BaseModel, Field

from app import settings
from app.schemas.asset_metadata.types import *
from app.schemas.enums import SupportedAssetType
from app.services.metadata_filtering.field_valid_values import get_field_valid_values

# README
# Value constraints need to be specified within their own annotated types rather than in the Field arguments
# Constraints passed via Field constructor are only applied to constrain the list size


class AutomaticallyExtractedMetadata(BaseModel):
    """
    Metadata fields that apply to any ML asset (dataset, model, software, publication, …)
    and are automatically extracted from the asset's JSON schema without the need to use an LLM.
    """

    platform: PlatformEnum | Varchar32 | None = Field(
        None,
        description=(
            "The platform on which the asset is hosted, e.g. 'Hugging Face', 'Zenodo', 'OpenML'."
        ),
    )
    name: Varchar256 | None = Field(
        None, description=("The name of the asset on the original platform.")
    )
    date_published: DatePublished | None = Field(
        None, description=("The date the asset was published on the original platform.")
    )
    same_as: Varchar256 | None = Field(
        None, description=("The link pointing to the asset on the original platform.")
    )

    @classmethod
    def get_date_field_names(cls) -> list[str]:
        return ["date_published"]

    @classmethod
    def get_categorical_field_names(cls) -> list[str]:
        return ["platform", "name", "same_as"]

    @classmethod
    def get_numerical_field_names(cls) -> list[str]:
        return []


class Base_AiExtractedMetadata(BaseModel):
    """
    Metadata fields that apply to any ML asset (dataset, model, software, publication, …)
    and which frequently need to be inferred from unstructured text.
    """

    research_areas: Optional[List[Varchar128]] = Field(
        None,
        description=(
            "Primary AI research disciplines, areas of research addressed by the asset, "
            "e.g. ['Natural Language Processing', 'Computer Vision', 'Reinforcement Learning']."
        ),
        max_length=16,
    )

    ml_tasks: Optional[List[Varchar64]] = Field(
        None,
        description=(
            "Core machine-learning tasks the asset tackles e.g. 'classification', 'segmentation', 'question-answering', "
            "'machine translation', etc. These task names align with common task categories used on the "
            "Hugging Face platform."
        ),
        max_length=64,
    )

    modalities: Optional[List[ModalityEnum]] = Field(
        None, description="Data modalities utilized within the asset.", max_length=16
    )

    languages: Optional[List[LanguageCode]] = Field(
        None,
        description=(
            "Human language(s) associated with the asset, using ISO-639-1 codes, "
            "e.g. ['en', 'de', 'fr']."
        ),
        max_length=128,
    )

    industrial_sectors: Optional[List[Varchar128]] = Field(
        None,
        description=(
            "Industry or economic sector(s) in which the asset is intended to be applied, e.g. "
            "['Manufacturing', 'Financial Services', 'Healthcare', "
            "'Agriculture', 'Public Administration']."
        ),
        max_length=16,
    )

    application_areas: Optional[List[Varchar128]] = Field(
        None,
        description=(
            "The practical application areas or use cases the asset addresses, e.g. "
            "['Predictive Maintenance', 'Fraud Detection', "
            "'Customer Churn Prevention', 'Supply-Chain Optimisation']."
        ),
        max_length=16,
    )

    license: Optional[Varchar32] = Field(
        None,
        description=("License associated with the asset, e.g. 'apache-2.0', 'cc-by-4.0', 'mit'."),
    )

    frameworks: Optional[List[Varchar128]] = Field(
        None,
        description=(
            "Software frameworks, libraries or standards referenced, e.g. ['PyTorch', 'TensorFlow', 'ONNX']."
        ),
        max_length=32,
    )

    @classmethod
    def get_date_field_names(cls) -> list[str]:
        return []

    @classmethod
    def get_categorical_field_names(cls) -> list[str]:
        # All the fields are categorical
        return list(cls.model_fields.keys())

    @classmethod
    def get_numerical_field_names(cls) -> list[str]:
        return []

    @classmethod
    def get_described_fields(cls) -> dict[str, str]:
        return {
            field_name: getattr(field, "description", "")
            for field_name, field in cls.model_fields.items()
        }


class AssetSpecific_AiExtractedMetadata(Base_AiExtractedMetadata):
    @classmethod
    @abstractmethod
    def get_asset_type(cls) -> SupportedAssetType:
        pass

    @classmethod
    @abstractmethod
    def get_date_field_names(cls) -> list[str]:
        pass

    @classmethod
    @abstractmethod
    def get_categorical_field_names(cls) -> list[str]:
        pass

    @classmethod
    @abstractmethod
    def get_numerical_field_names(cls) -> list[str]:
        pass

    @classmethod
    def get_described_fields(cls) -> dict[str, str]:
        return {
            field_name: getattr(field, "description", "")
            for field_name, field in cls.model_fields.items()
        }


class AssetSpecific_UserQueryParsedMetadata(AutomaticallyExtractedMetadata):
    @classmethod
    @abstractmethod
    def get_asset_type(cls) -> SupportedAssetType:
        pass

    @classmethod
    def get_described_fields(cls) -> dict[str, str]:
        return {
            field_name: getattr(field, "description", "")
            for field_name, field in cls.model_fields.items()
        }

    @classmethod
    def get_inner_annotation(cls, field_name: str, with_valid_values_enum: bool) -> type:
        all_valid_values = get_field_valid_values().get_values(cls.get_asset_type(), field_name)

        if (
            with_valid_values_enum
            and all_valid_values
            and settings.METADATA_FILTERING.ENFORCE_ENUMS
        ):
            return cast(type, Literal[*all_valid_values])
        else:
            annotation = cls._get_annotation_or_raise(field_name)
            return AnnotationOperations.strip_optional_and_list_types(annotation)

    @classmethod
    def get_list_fields_mask(cls) -> dict[str, bool]:
        return {
            field_name: AnnotationOperations.is_list_type(
                AnnotationOperations.strip_optional_type(cls._get_annotation_or_raise(field_name))
            )
            for field_name in cls.model_fields.keys()
        }

    @classmethod
    def _get_annotation_or_raise(cls, field_name: str) -> type:
        annotation = cls.model_fields[field_name].annotation
        if annotation is None:
            raise ValueError(
                f"Annotation for the field '{field_name}' of the model '{cls.__name__}' doesn't exist. Fix the asset schema."
            )
        else:
            return annotation

    @classmethod
    def get_supported_comparison_operators(cls, field_name: str) -> list[ComparisonOperator]:
        match_operators: list[ComparisonOperator] = ["==", "!="]
        range_operators: list[ComparisonOperator] = [">", "<", ">=", "<="]

        if field_name in cls.get_date_field_names():
            return range_operators
        elif field_name in cls.get_categorical_field_names():
            return match_operators
        elif field_name in cls.get_numerical_field_names():
            return match_operators + range_operators
        else:
            raise ValueError(f"Invalid field name: '{field_name}'")

    @classmethod
    @abstractmethod
    def get_date_field_names(cls) -> list[str]:
        pass

    @classmethod
    @abstractmethod
    def get_categorical_field_names(cls) -> list[str]:
        pass

    @classmethod
    @abstractmethod
    def get_numerical_field_names(cls) -> list[str]:
        pass
