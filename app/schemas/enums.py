from __future__ import annotations

from enum import Enum


class QueryStatus(Enum):
    QUEUED = "Queued"
    IN_PROGRESS = "In_progress"
    COMPLETED = "Completed"
    FAILED = "Failed"


class SupportedAssetType(Enum):
    DATASETS = "datasets"
    ML_MODELS = "ml_models"
    PUBLICATIONS = "publications"
    CASE_STUDIES = "case_studies"
    EDUCATIONAL_RESOURCES = "educational_resources"
    EXPERIMENTS = "experiments"
    SERVICES = "services"

    def to_SupportedAssetType(self) -> SupportedAssetType:
        return self

    def is_all(self) -> bool:
        return False


# TODO come up with a way to get rid of this duplication of enum values
class AssetTypeQueryParam(Enum):
    ALL = "all"  # select all asset types
    DATASETS = "datasets"
    ML_MODELS = "ml_models"
    PUBLICATIONS = "publications"
    CASE_STUDIES = "case_studies"
    EDUCATIONAL_RESOURCES = "educational_resources"
    EXPERIMENTS = "experiments"
    SERVICES = "services"

    def to_SupportedAssetType(self) -> SupportedAssetType:
        if self == AssetTypeQueryParam.ALL:
            raise ValueError("ALL value cannot be converted")
        return SupportedAssetType(self.value)

    def is_all(self) -> bool:
        return self == AssetTypeQueryParam.ALL
