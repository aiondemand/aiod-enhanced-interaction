import json
from pathlib import Path
from typing import Annotated, Any, ClassVar, Optional, Type

from app.schemas.asset_metadata.base import BaseInnerAnnotations, BaseMetadataTemplate
from pydantic import Field, field_validator

# Every metadata field we use for filtering purposes has 2 annotations associated with it:
# - The inner annotation corresponding to a singular eligible value for the field;
#   - This annotation is used for creating and validating user queries and filters/conditions
# - The outer annotation wrapping the inner annotation and extending it by allowing for a list of values if necessary
#   - This annotation is used for extracting metadata from AIoD assets automatically using an LLM


class DatasetInnerAnnotations(BaseInnerAnnotations):
    @staticmethod
    def _load_all_valid_values(path: Path) -> dict[str, list[str]]:
        with open(path) as f:
            valid_values = json.load(f)
        return valid_values

    _ALL_VALID_VALUES: ClassVar[dict[str, list[str]]] = _load_all_valid_values(
        Path("data/valid_metadata_values.json")
    )

    @classmethod
    def exists_list_of_valid_values(cls, field_name: str) -> bool:
        return field_name in cls._ALL_VALID_VALUES

    @classmethod
    def get_list_of_valid_values(cls, field_name: str) -> list[str]:
        if not cls.exists_list_of_valid_values(field_name):
            raise ValueError(f"Field {field_name} does not have a list of valid values")
        return cls._ALL_VALID_VALUES[field_name]

    DatePublished = Annotated[
        str,
        Field(
            description="The publication date of the dataset in the format 'YYYY-MM-DDTHH:MM:SSZ'.",
            pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$",
        ),
    ]
    SizeInMb = Annotated[
        int, Field(description="The total size of the dataset in megabytes.", ge=0)
    ]
    License = Annotated[
        str,
        Field(
            description="The license of the dataset. Only a subset of licenses are recognized.",
            enum=_ALL_VALID_VALUES["license"],
        ),
    ]
    TaskType = Annotated[
        str,
        Field(
            description="The machine learning tasks corresponding to this dataset. Only a subset of task types are recognized.",
            enum=_ALL_VALID_VALUES["task_types"],
        ),
    ]
    Language = Annotated[
        str,
        Field(
            description="The language of the dataset specified as an ISO 639-1 two-letter code.",
            min_length=2,
            max_length=2,
        ),
    ]
    DatapointsLowerBound = Annotated[
        int, Field(description="The lower bound of the number of datapoints in the dataset.", ge=0)
    ]
    DatapointsUpperBound = Annotated[
        int, Field(description="The upper bound of the number of datapoints in the dataset.", ge=0)
    ]


class HuggingFaceDatasetMetadataTemplate(BaseMetadataTemplate[DatasetInnerAnnotations]):
    """
    Extraction of relevant metadata we wish to retrieve from ML assets
    """

    date_published: Optional[DatasetInnerAnnotations.DatePublished] = Field(
        None,
        description="The publication date of the dataset in the format 'YYYY-MM-DDTHH:MM:SSZ'. Don't forget to convert the date to appropriate format if necessary.",
    )
    size_in_mb: Optional[DatasetInnerAnnotations.SizeInMb] = Field(
        None,
        description="The total size of the dataset in megabytes. Don't forget to convert the sizes to MBs if necessary.",
    )
    license: Optional[DatasetInnerAnnotations.License] = Field(
        None, description="The license of the dataset, e.g., 'mit', 'apache'"
    )
    task_types: Optional[list[DatasetInnerAnnotations.TaskType]] = Field(
        None,
        description="The machine learning tasks suitable for this dataset. Acceptable values may include task categories or task ids found on HuggingFace platform (e.g., 'token-classification', 'question-answering', ...)",
    )
    languages: Optional[list[DatasetInnerAnnotations.Language]] = Field(
        None,
        description="Languages present in the dataset, specified in ISO 639-1 two-letter codes (e.g., 'en' for English, 'es' for Spanish, 'fr' for French, etc ...).",
    )
    datapoints_lower_bound: Optional[DatasetInnerAnnotations.DatapointsLowerBound] = Field(
        None,
        description="The lower bound of the number of datapoints in the dataset. This value represents the minimum number of datapoints found in the dataset.",
    )
    datapoints_upper_bound: Optional[DatasetInnerAnnotations.DatapointsUpperBound] = Field(
        None,
        description="The upper bound of the number of datapoints in the dataset. This value represents the maximum number of datapoints found in the dataset.",
    )

    @field_validator("license", "task_types", "languages", mode="before")
    @classmethod
    def convert_strings_to_lowercase(cls, value: Any) -> Any:
        return cls.apply_string_function_recursively(value, str.lower)

    @field_validator("date_published", mode="before")
    @classmethod
    def convert_strings_to_uppercase(cls, value: Any) -> Any:
        return cls.apply_string_function_recursively(value, str.upper)

    # FieldInfo enum argument is only for documentation purposes (OpenAPI schema generation)
    # It doesn't affect the validation logic, hence the need to check the fields manually
    @field_validator("license", mode="before")
    @classmethod
    def check_license(cls, value: str) -> str | None:
        valid_values = DatasetInnerAnnotations.get_list_of_valid_values("license")
        return value if value in valid_values else None

    @field_validator("task_types", mode="before")
    @classmethod
    def check_task_types(cls, values: list[str]) -> list[str] | None:
        valid_values = DatasetInnerAnnotations.get_list_of_valid_values("task_types")
        return [val for val in values if val.lower() in valid_values]

    @classmethod
    def get_inner_annotations_class(cls) -> Type[DatasetInnerAnnotations]:
        return DatasetInnerAnnotations

    @classmethod
    def get_supported_comparison_operators(cls, field_name: str) -> list[str]:
        match_operators = ["==", "!="]
        range_operators = [">", "<", ">=", "<="]

        if field_name in ["date_published"]:
            return range_operators
        elif field_name in ["license", "task_types", "languages"]:
            return match_operators
        elif field_name in ["size_in_mb", "datapoints_lower_bound", "datapoints_upper_bound"]:
            return match_operators + range_operators
        else:
            raise ValueError(f"Invalid field name: {field_name}")
