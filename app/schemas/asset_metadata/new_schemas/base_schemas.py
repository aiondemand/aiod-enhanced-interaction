from typing import List, Optional
from pydantic import BaseModel, Field

from app.schemas.asset_metadata.new_schemas.types import (
    DatePublished,
    LanguageCode,
    ModalityEnum,
    Varchar128,
    Varchar256,
    Varchar32,
    Varchar64,
)


# TODO LATER
# Add into the Pydantic models value constraints matching the ones we use for building
# Milvus collections (for each string field => max_length, max_capacity)

# Value constraints need to be specified within their own annotated types rather than in the Field arguments
# Constraints passed via Field constructor are only applied to constrain the list size


class AutomaticallyExtractedMetadata(BaseModel):
    """
    Metadata fields that apply to any ML asset (dataset, model, software, publication, …)
    and are automatically extracted from the asset's JSON schema without the need to use an LLM.
    """

    platform: Varchar32 | None = Field(
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


class AssetSpecificMetadata(Base_AiExtractedMetadata):
    @classmethod
    def get_described_fields(cls) -> dict[str, str]:
        return {
            field_name: getattr(field, "description", "")
            for field_name, field in cls.model_fields.items()
        }

    pass
