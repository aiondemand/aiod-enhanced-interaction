from typing import List, Optional

from pydantic import Field

from app.schemas.asset_metadata.base_schemas import (
    AssetSpecific_AiExtractedMetadata,
    AssetSpecific_UserQueryParsedMetadata,
    AutomaticallyExtractedMetadata,
    Base_AiExtractedMetadata,
)
from app.schemas.asset_metadata.types import *


class EducationalResource_AiExtractedMetadata(AssetSpecific_AiExtractedMetadata):
    """
    Metadata fields that apply only to assets of type 'educational_resource'.
    Every attribute is optional so an agent can omit values it cannot
    infer with confidence.
    """

    resource_types: Optional[List[EducationalResourceType]] = Field(
        None,
        description=(
            "Format of the learning material, e.g. 'video recordings', 'book', 'dataset'."
        ),
        max_length=8,
    )

    educational_levels: Optional[List[Varchar128]] = Field(
        None,
        description=(
            "Formal education stage the material targets, e.g. 'Bachelor’s or equivalent level'."
        ),
        max_length=8,
    )

    competency_levels: Optional[List[EducationalCompetency]] = Field(
        None,
        description=(
            "Expected learner proficiency such as 'beginner', 'intermediate', or 'advanced'."
        ),
        max_length=4,
    )

    learning_modes: Optional[List[LearningMode]] = Field(
        None,
        description=("Primary delivery mode for the resource, e.g. online, offline, or hybrid."),
        max_length=4,
    )

    educational_paces: Optional[List[EducationalPace]] = Field(
        None,
        description=("Whether the learning is self-paced, scheduled, or full-time."),
        max_length=4,
    )

    prerequisites: Optional[List[Varchar128]] = Field(
        None,
        description=(
            "Pre-existing knowledge or skills recommended before engaging with the material, using predefined prerequisite categories."
        ),
        max_length=16,
    )

    target_audiences: Optional[List[EducationalTargetAudience]] = Field(
        None,
        description=(
            "Intended learner groups, e.g. professionals, working professionals, or students."
        ),
        max_length=16,
    )

    estimated_duration_hours: Optional[float] = Field(
        None,
        description=("Approximate time commitment required to complete the material, in hours."),
    )

    provider: Optional[Varchar128] = Field(
        None,
        description=("Organization or platform offering the learning resource."),
    )

    certification_available: Optional[bool] = Field(
        None,
        description=("True if learners can obtain a certificate or badge after completion."),
    )

    @classmethod
    def get_date_field_names(cls) -> list[str]:
        return Base_AiExtractedMetadata.get_date_field_names() + []

    @classmethod
    def get_categorical_field_names(cls) -> list[str]:
        return Base_AiExtractedMetadata.get_categorical_field_names() + [
            "resource_types",
            "educational_levels",
            "competency_levels",
            "learning_modes",
            "educational_paces",
            "prerequisites",
            "target_audiences",
            "provider",
            "certification_available",
        ]

    @classmethod
    def get_numerical_field_names(cls) -> list[str]:
        return Base_AiExtractedMetadata.get_numerical_field_names() + ["estimated_duration_hours"]


class EducationalResource_UserQueryParsedMetadata(
    AssetSpecific_UserQueryParsedMetadata, EducationalResource_AiExtractedMetadata
):
    @classmethod
    def get_date_field_names(cls) -> list[str]:
        return (
            AutomaticallyExtractedMetadata.get_date_field_names()
            + EducationalResource_AiExtractedMetadata.get_date_field_names()
        )

    @classmethod
    def get_categorical_field_names(cls) -> list[str]:
        return (
            AutomaticallyExtractedMetadata.get_categorical_field_names()
            + EducationalResource_AiExtractedMetadata.get_categorical_field_names()
        )

    @classmethod
    def get_numerical_field_names(cls) -> list[str]:
        return (
            AutomaticallyExtractedMetadata.get_numerical_field_names()
            + EducationalResource_AiExtractedMetadata.get_numerical_field_names()
        )
