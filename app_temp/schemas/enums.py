from __future__ import annotations

from enum import Enum


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
