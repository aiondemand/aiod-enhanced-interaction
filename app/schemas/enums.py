from __future__ import annotations

from enum import Enum


class QueryStatus(Enum):
    QUEUED = "Queued"
    IN_PROGRESS = "In_progress"
    COMPLETED = "Completed"
    FAILED = "Failed"


class BaseAssetType:
    def to_SupportedAssetType(self) -> SupportedAssetType:
        if self.is_all():
            raise ValueError("ALL value cannot be converted")
        return SupportedAssetType(getattr(self, "value"))

    def is_all(self) -> bool:
        return getattr(self, "value") == "all"


class SupportedAssetType(BaseAssetType, Enum):
    DATASETS = "datasets"
    ML_MODELS = "ml_models"
    PUBLICATIONS = "publications"
    CASE_STUDIES = "case_studies"
    EDUCATIONAL_RESOURCES = "educational_resources"
    EXPERIMENTS = "experiments"
    SERVICES = "services"

    # def to_SupportedAssetType(self) -> SupportedAssetType:
    #     return self

    # def is_all(self) -> bool:
    #     return False


# TODO come up with a way to get rid of this duplication of enum values
class AssetTypeQueryParam(BaseAssetType, Enum):
    ALL = "all"  # select all asset types
    DATASETS = "datasets"
    ML_MODELS = "ml_models"
    PUBLICATIONS = "publications"
    CASE_STUDIES = "case_studies"
    EDUCATIONAL_RESOURCES = "educational_resources"
    EXPERIMENTS = "experiments"
    SERVICES = "services"

    # def to_SupportedAssetType(self) -> SupportedAssetType:
    #     if self == AssetTypeQueryParam.ALL:
    #         raise ValueError("ALL value cannot be converted")
    #     return SupportedAssetType(self.value)

    # def is_all(self) -> bool:
    #     return self == AssetTypeQueryParam.ALL
