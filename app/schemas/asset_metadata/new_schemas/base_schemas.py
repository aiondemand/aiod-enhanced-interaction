from typing import List, Optional
from pydantic import BaseModel, Field

from app.schemas.asset_metadata.new_schemas.types import DatePublished, LanguageCode, ModalityEnum


# TODO LATER
# Add into the Pydantic models value constraints matching the ones we use for building
# Milvus collections (for each string field => max_length, max_capacity)


class AutomaticallyExtractedMetadata(BaseModel):
    """
    Metadata fields that apply to any ML asset (dataset, model, software, publication, …)
    and are automatically extracted from the asset's JSON schema without the need to use an LLM.
    """

    platform: str | None = Field(
        None,
        description=(
            "The platform on which the asset is hosted, e.g. 'Hugging Face', 'Zenodo', 'OpenML'."
        ),
    )
    name: str | None = Field(None, description=("The name of the asset on the original platform."))
    date_published: DatePublished | None = Field(
        None, description=("The date the asset was published on the original platform.")
    )
    same_as: str | None = Field(
        None, description=("The link pointing to the asset on the original platform.")
    )


class Base_AiExtractedMetadata(BaseModel):
    """
    Metadata fields that apply to any ML asset (dataset, model, software, publication, …)
    and which frequently need to be inferred from unstructured text.
    """

    research_areas: Optional[List[str]] = Field(
        None,
        description=(
            "Primary AI research disciplines, areas of research addressed by the asset, "
            "e.g. ['Natural Language Processing', 'Computer Vision', 'Reinforcement Learning']."
        ),
    )

    ml_tasks: Optional[List[str]] = Field(
        None,
        description=(
            "Core machine-learning tasks the asset tackles e.g. 'classification', 'segmentation', 'question-answering', "
            "'machine translation', etc. These task names align with common task categories used on the "
            "Hugging Face platform."
        ),
    )

    modalities: Optional[List[ModalityEnum]] = Field(
        None, description="Data modalities utilized within the asset."
    )

    languages: Optional[List[LanguageCode]] = Field(
        None,
        description=(
            "Human language(s) associated with the asset, using ISO-639-1 codes, "
            "e.g. ['en', 'de', 'fr']."
        ),
    )

    industrial_sectors: Optional[List[str]] = Field(
        None,
        description=(
            "Industry or economic sector(s) in which the asset is intended to be applied, e.g. "
            "['Manufacturing', 'Financial Services', 'Healthcare', "
            "'Agriculture', 'Public Administration']."
        ),
    )

    application_areas: Optional[List[str]] = Field(
        None,
        description=(
            "The practical application areas or use cases the asset addresses, e.g. "
            "['Predictive Maintenance', 'Fraud Detection', "
            "'Customer Churn Prevention', 'Supply-Chain Optimisation']."
        ),
    )

    license: Optional[str] = Field(
        None,
        description=("License associated with the asset, e.g. 'apache-2.0', 'cc-by-4.0', 'mit'."),
    )

    frameworks: Optional[List[str]] = Field(
        None,
        description=(
            "Software frameworks, libraries or standards referenced, e.g. ['PyTorch', 'TensorFlow', 'ONNX']."
        ),
    )


class AssetSpecificMetadata(Base_AiExtractedMetadata):
    @classmethod
    def get_described_fields(cls) -> dict[str, str]:
        return {
            field_name: getattr(field, "description", "")
            for field_name, field in cls.model_fields.items()
        }

    pass
