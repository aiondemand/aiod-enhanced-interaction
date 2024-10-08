from enum import Enum


class QueryStatus(Enum):
    QUEUED = "Queued"
    IN_PROGESS = "In_progress"
    COMPLETED = "Completed"


class AssetType(Enum):
    DATASETS = "datasets"
    ML_MODELS = "ml_models"
    PUBLICATIONS = "publications"
    CASE_STUDIES = "case_studies"
    EDUCATIONAL_RESOURCES = "educational_resources"
    EXPERIMENTS = "experiments"
