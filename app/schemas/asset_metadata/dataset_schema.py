from typing import List, Optional
from pydantic import Field

from app.schemas.asset_metadata.base_schemas import (
    AssetSpecific_AiExtractedMetadata,
    AssetSpecific_UserQueryParsedMetadata,
    AutomaticallyExtractedMetadata,
    Base_AiExtractedMetadata,
)
from app.schemas.asset_metadata.types import *
from app.schemas.enums import SupportedAssetType


class Dataset_AiExtractedMetadata(AssetSpecific_AiExtractedMetadata):
    """
    Metadata fields that apply only to assets of type 'dataset'.
    Every attribute is optional so an agent can omit values it cannot
    infer with confidence.
    """

    datapoint_count: Optional[int] = Field(
        None,
        description=(
            "Approximate number of rows / instances / datapoints in the dataset, e.g. 120 000."
        ),
    )

    feature_count: Optional[int] = Field(
        None,
        description=("Number of features / columns per instance / datapoint, e.g. 42."),
    )

    data_formats: Optional[List[FileExtension]] = Field(
        None,
        description=(
            "One or more storage formats specified by their extension e.g. '.png', '.mp3', '.csv', '.json', '.parquet', etc."
        ),
        max_length=16,
    )

    label_types: Optional[List[Varchar32]] = Field(
        None,
        description=(
            "Kinds of annotations, label types  present , e.g. "
            "['single_label_class', 'numeric_scalar', 'sequence_label', "
            "'text_generation', 'bounding_box', 'segmentation_mask', "
            "'keypoints', 'pairwise_rank', 'ordered_rank']."
        ),
        max_length=8,
    )

    collection_methods: Optional[List[Varchar32]] = Field(
        None,
        description=(
            "How the raw data was gathered, e.g. "
            "['sensor', 'survey', 'web_scraping', 'simulation', 'synthetic_generation'"
            "'crowdsourcing', 'third_party', 'internal_systems']."
        ),
        max_length=8,
    )

    source_type: Optional[SourceTypeEnum] = Field(
        None, description="What source the data comes from"
    )

    update_frequency: Optional[Varchar32] = Field(
        None,
        description=(
            "How often the dataset is updated, e.g. "
            "'never', 'on_demand', 'daily', 'weekly', 'monthly', "
            "'quarterly', 'annually'"
        ),
    )

    dataset_size_gigabytes: Optional[float] = Field(
        None,
        description=("Total compressed size of the dataset files in gigabytes"),
    )

    class_count: Optional[int] = Field(
        None,
        description=("Number of unique classes / labels (for classification datasets), e.g. 10."),
    )

    geo_coverage: Optional[List[CountryCode]] = Field(
        None,
        description=(
            "List of ISO-3166 country codes or region names represented in the data, e.g. ['US', 'DE', 'CN']."
        ),
        max_length=64,
    )

    temporal_coverage_start: Optional[DateString] = Field(
        None,
        description=("Start date of temporal coverage in the dataset."),
    )

    temporal_coverage_end: Optional[DateString] = Field(
        None,
        description=("End date of temporal coverage in the dataset."),
    )

    @classmethod
    def get_asset_type(cls) -> SupportedAssetType:
        return SupportedAssetType.DATASETS

    @classmethod
    def get_date_field_names(cls) -> list[str]:
        return Base_AiExtractedMetadata.get_date_field_names() + [
            "temporal_coverage_start",
            "temporal_coverage_end",
        ]

    @classmethod
    def get_categorical_field_names(cls) -> list[str]:
        return Base_AiExtractedMetadata.get_categorical_field_names() + [
            "data_formats",
            "label_types",
            "collection_methods",
            "source_type",
            "update_frequency",
            "geo_coverage",
        ]

    @classmethod
    def get_numerical_field_names(cls) -> list[str]:
        return Base_AiExtractedMetadata.get_numerical_field_names() + [
            "datapoint_count",
            "feature_count",
            "dataset_size_gigabytes",
            "class_count",
        ]


class Dataset_UserQueryParsedMetadata(
    AssetSpecific_UserQueryParsedMetadata, Dataset_AiExtractedMetadata
):
    @classmethod
    def get_asset_type(cls) -> SupportedAssetType:
        return SupportedAssetType.DATASETS

    @classmethod
    def get_date_field_names(cls) -> list[str]:
        return (
            AutomaticallyExtractedMetadata.get_date_field_names()
            + Dataset_AiExtractedMetadata.get_date_field_names()
        )

    @classmethod
    def get_categorical_field_names(cls) -> list[str]:
        return (
            AutomaticallyExtractedMetadata.get_categorical_field_names()
            + Dataset_AiExtractedMetadata.get_categorical_field_names()
        )

    @classmethod
    def get_numerical_field_names(cls) -> list[str]:
        return (
            AutomaticallyExtractedMetadata.get_numerical_field_names()
            + Dataset_AiExtractedMetadata.get_numerical_field_names()
        )
