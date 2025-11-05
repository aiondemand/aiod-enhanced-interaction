from typing import List, Optional
from pydantic import BaseModel, Field

from app.schemas.asset_metadata.new_schemas.enums import ModalityEnum


class AutomaticallyExtractedMetadata(BaseModel):
    """
    Metadata fields that apply to any ML asset (dataset, model, software, publication, …)
    and are automatically extracted from the asset's JSON schema without the need to use an LLM.
    """

    platform: str | None = Field(
        ...,
        description=(
            "The platform on which the asset is hosted, e.g. 'Hugging Face', 'Zenodo', 'OpenML'."
        ),
    )
    name: str | None = Field(..., description=("The name of the asset on the original platform."))
    date_published: str | None = Field(
        ..., description=("The date the asset was published on the original platform.")
    )
    same_as: str | None = Field(
        ..., description=("The link pointing to the asset on the original platform.")
    )


class Base_AiExtractedMetadata(BaseModel):
    """
    Metadata fields that apply to any ML asset (dataset, model, software, publication, …)
    and which frequently need to be inferred from unstructured text.
    """

    # Exists a AIoD taxonomy (Research Area) for this field
    research_areas: Optional[List[str]] = Field(
        None,
        description=(
            "Primary AI research disciplines, areas of research addressed by the asset, "
            "e.g. ['Natural Language Processing', 'Computer Vision', 'Reinforcement Learning']."
        ),
    )

    # TODO We need to come up with a enum for this field
    ml_tasks: Optional[List[str]] = Field(
        None,
        description=(
            "Core machine-learning task the asset tackles e.g. 'classification', 'segmentation', 'question-answering', "
            "'machine translation', etc. These task names align with common task categories used on the "
            "Hugging Face platform."
        ),
    )

    modalities: Optional[List[ModalityEnum]] = Field(
        None,
        description=(
            "Data modalities utilized within the asset, e.g. ['text'], ['image', 'text'], ['audio', 'video']."
        ),
    )

    languages: Optional[List[str]] = Field(
        None,
        description=(
            "Human language(s) associated with the asset, using ISO-639-1/2 codes, "
            "e.g. ['en', 'de', 'fr']."
        ),
    )

    # TODO
    # Exists a AIoD taxonomy (Business Sector) for this field
    bussiness_sectors: Optional[List[str]] = Field(
        None,
        description=(
            "Industry or economic sector(s) in which the asset is intended to be applied, e.g. "
            "['Manufacturing', 'Financial Services', 'Healthcare', "
            "'Agriculture', 'Public Administration']."
        ),
    )

    # TODO
    # Exists a taxonomy (Business Problem) for this field
    business_problems: Optional[List[str]] = Field(
        None,
        description=(
            "The specific business problem(s) the asset aims to solve, e.g. "
            "['Predictive Maintenance', 'Fraud Detection', "
            "'Customer Churn Prevention', 'Supply-Chain Optimisation']."
        ),
    )

    # Exists a AIoD taxonomy (License) for this field
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

    author_affiliations: Optional[List[str]] = Field(
        None,
        description=(
            "Institutional affiliations of the main authors/maintainers, e.g. "
            "['Fraunhofer IAIS', 'University College Cork']."
        ),
    )

    # TODO for publications and educational resources
    # TODO
    # Exists a AIoD taxonomy (Publication) for this field
    # publication_type: Optional[str] = Field(
    #     None,
    #     description=(
    #         "Type of publication the asset is, e.g. 'book', 'chapter', 'journal_article', "
    #         "'conference_paper', 'preprint', 'blog'"
    #     ),
    # )


class AssetSpecificMetadata(Base_AiExtractedMetadata):
    pass
