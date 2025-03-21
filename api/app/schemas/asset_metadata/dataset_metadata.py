import json
import re
from pathlib import Path
from typing import ClassVar, Optional

from pydantic import BaseModel, Field, field_validator

# Rules how to set up these schemas representing asset metadata
#   1. Each pydantic.Field only contains a default value and a description.
#      Other arguments are not copied over when creating dynamic schemas.
#   2. If you wish to apply some additional value constraints, feel free to do so, but
#      you're expected to apply them in a corresponding field_validator instead.


class HuggingFaceDatasetMetadataTemplate(BaseModel):
    """
    Extraction of relevant metadata we wish to retrieve from ML assets
    """

    @staticmethod
    def _load_all_valid_values(path: Path) -> dict[str, list[str]]:
        with open(path) as f:
            valid_values = json.load(f)
        return valid_values

    _ALL_VALID_VALUES: ClassVar[dict[str, list[str]]] = _load_all_valid_values(
        Path("data/valid_metadata_values.json")
    )

    date_published: Optional[str] = Field(
        None,
        description="The publication date of the dataset in the format 'YYYY-MM-DDTHH:MM:SSZ'.",
    )
    size_in_mb: Optional[int] = Field(
        None,
        description="The total size of the dataset in megabytes. Don't forget to convert the sizes to MBs if necessary.",
    )
    license: Optional[str] = Field(
        None,
        description="The license associated with this dataset, e.g., 'mit', 'apache'",
    )
    task_types: Optional[list[str]] = Field(
        None,
        description="The machine learning tasks suitable for this dataset. Acceptable values may include task categories or task ids found on HuggingFace platform (e.g., 'token-classification', 'question-answering', ...)",
    )
    languages: Optional[list[str]] = Field(
        None,
        description="Languages present in the dataset, specified in ISO 639-1 two-letter codes (e.g., 'en' for English, 'es' for Spanish, 'fr' for French, etc ...).",
    )
    datapoints_lower_bound: Optional[int] = Field(
        None,
        description="The lower bound of the number of datapoints in the dataset. This value represents the minimum number of datapoints found in the dataset.",
    )
    datapoints_upper_bound: Optional[int] = Field(
        None,
        description="The upper bound of the number of datapoints in the dataset. This value represents the maximum number of datapoints found in the dataset.",
    )

    @classmethod
    def get_field_valid_values(cls, field: str) -> list[str]:
        return cls._ALL_VALID_VALUES.get(field, [])

    @classmethod
    def exists_field_valid_values(cls, field: str) -> bool:
        return field in cls._ALL_VALID_VALUES.keys()

    @classmethod
    @field_validator("date_published", mode="before")
    def check_date_published(cls, value: str) -> str | None:
        pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
        return value if bool(re.match(pattern, value)) else None

    @classmethod
    @field_validator("license", mode="before")
    def check_license(cls, value: str) -> str | None:
        return value if value in cls.get_field_valid_values("license") else None

    @classmethod
    @field_validator("task_types", mode="before")
    def check_task_types(cls, values: list[str]) -> list[str] | None:
        valid_values = [
            val.lower() for val in values if val.lower() in cls.get_field_valid_values("task_types")
        ]
        return valid_values

    @classmethod
    @field_validator("languages", mode="before")
    def check_languages(cls, values: list[str]) -> list[str] | None:
        return [val.lower() for val in values if len(val) == 2]
