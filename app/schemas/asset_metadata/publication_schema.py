from typing import List, Optional

from pydantic import Field

from app.schemas.asset_metadata.base_schemas import *
from app.schemas.asset_metadata.types import *
from app.schemas.enums import SupportedAssetType


class Publication_AiExtractedMetadata(AssetSpecific_AiExtractedMetadata):
    """
    Metadata fields that apply only to assets of type 'publication'.
    Every attribute is optional so an agent can omit values it cannot
    infer with confidence.
    """

    authors: Optional[List[Varchar128]] = Field(
        None,
        description=(
            "Ordered list of authors credited on the work, formatted as they appear in the publication."
        ),
        max_length=64,
    )

    affiliations: Optional[List[Varchar128]] = Field(
        None,
        description=(
            "Institutions or organizations the authors are affiliated with, e.g. universities, labs, companies."
        ),
        max_length=32,
    )

    publication_types: Optional[List[Varchar128]] = Field(
        None,
        description=(
            "Document format or venue category of the work, e.g. 'Article', 'Conference Poster', 'Report Document'."
        ),
        max_length=8,
    )

    publication_venues: Optional[List[Varchar128]] = Field(
        None,
        description=(
            "Name of the outlet where the work appeared, e.g. journal, conference, workshop, preprint server."
        ),
        max_length=4,
    )

    news_categories: Optional[List[NewsCategory]] = Field(
        None,
        description=(
            "High-level topic area if the item is a news-style publication, e.g. 'Research'."
        ),
        max_length=4,
    )

    peer_reviewed: Optional[bool] = Field(
        None,
        description=("True if the work underwent formal peer review before publication."),
    )

    open_access: Optional[bool] = Field(
        None,
        description=(
            "True if the full text is openly accessible without paywalls or registration."
        ),
    )

    doi: Optional[Varchar128] = Field(
        None,
        description=("Digital Object Identifier string such as '10.48550/arxiv.2401.12345'."),
    )

    citation_count: Optional[int] = Field(
        None,
        description=("Approximate number of times the work has been cited by other publications."),
    )

    @classmethod
    def get_asset_type(cls) -> SupportedAssetType:
        return SupportedAssetType.PUBLICATIONS

    @classmethod
    def get_date_field_names(cls) -> list[str]:
        return Base_AiExtractedMetadata.get_date_field_names() + []

    @classmethod
    def get_categorical_field_names(cls) -> list[str]:
        return Base_AiExtractedMetadata.get_categorical_field_names() + [
            "authors",
            "affiliations",
            "publication_types",
            "publication_venues",
            "news_categories",
            "peer_reviewed",
            "open_access",
            "doi",
        ]

    @classmethod
    def get_numerical_field_names(cls) -> list[str]:
        return Base_AiExtractedMetadata.get_numerical_field_names() + ["citation_count"]


class Publication_UserQueryParsedMetadata(
    AssetSpecific_UserQueryParsedMetadata, Publication_AiExtractedMetadata
):
    @classmethod
    def get_asset_type(cls) -> SupportedAssetType:
        return SupportedAssetType.PUBLICATIONS

    @classmethod
    def get_date_field_names(cls) -> list[str]:
        return (
            AutomaticallyExtractedMetadata.get_date_field_names()
            + Publication_AiExtractedMetadata.get_date_field_names()
        )

    @classmethod
    def get_categorical_field_names(cls) -> list[str]:
        return (
            AutomaticallyExtractedMetadata.get_categorical_field_names()
            + Publication_AiExtractedMetadata.get_categorical_field_names()
        )

    @classmethod
    def get_numerical_field_names(cls) -> list[str]:
        return (
            AutomaticallyExtractedMetadata.get_numerical_field_names()
            + Publication_AiExtractedMetadata.get_numerical_field_names()
        )
