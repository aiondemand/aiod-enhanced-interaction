from typing import List, Optional
from pydantic import Field

from app.schemas.asset_metadata.new_schemas.base_schemas import AssetSpecificMetadata


class Dataset_AiExtractedMetadata(AssetSpecificMetadata):
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

    data_formats: Optional[List[str]] = Field(
        None,
        description=(
            "One or more storage formats e.g. '.png', '.mp3', '.csv', '.json', '.parquet', etc."
        ),
    )

    # TODO incorporate LabelTypeEnum from enums.py
    label_types: Optional[List[str]] = Field(
        None,
        description=(
            "Kinds of annotations, label types  present , e.g. "
            "['single_label_class', 'numeric_scalar', 'sequence_label', "
            "'text_generation', 'bounding_box', 'segmentation_mask', "
            "'keypoints', 'pairwise_rank', 'ordered_rank']."
        ),
    )

    # TODO We need to come up with a enum for this field
    collection_methods: Optional[List[str]] = Field(
        None,
        description=(
            "How the raw data was gathered, e.g. "
            "['sensor', 'survey', 'web_scraping', 'simulation', 'synthetic_generation'"
            "'crowdsourcing', 'third_party', 'internal_systems']."
        ),
    )

    # TODO We need to come up with a enum for this field
    source_type: Optional[str] = Field(
        None,
        description=(
            "What source the data comes from, e.g. "
            "['real_world', 'synthetic', 'simulated', 'hybrid']."
        ),
    )

    # TODO We need to come up with a enum for this field
    update_frequency: Optional[str] = Field(
        None,
        description=(
            "How often the dataset is updated, e.g. "
            "['never', 'on_demand', 'daily', 'weekly', 'monthly', "
            "'quarterly', 'annually']."
        ),
    )

    dataset_size_megabytes: Optional[int] = Field(
        None,
        description=("Total compressed size of the dataset files in megabytes, e.g. 4700."),
    )

    class_count: Optional[int] = Field(
        None,
        description=("Number of unique classes / labels (for classification datasets), e.g. 10."),
    )

    geo_coverage: Optional[List[str]] = Field(
        None,
        description=(
            "List of ISO-3166 country codes or region names represented in the data, e.g. ['US', 'DE', 'CN']."
        ),
    )

    temporal_coverage_start: Optional[str] = Field(
        None,
        description=("Start date of temporal coverage in the dataset (YYYY-MM-DD format)."),
    )

    temporal_coverage_end: Optional[str] = Field(
        None,
        description=("End date of temporal coverage in the dataset (YYYY-MM-DD format)."),
    )
