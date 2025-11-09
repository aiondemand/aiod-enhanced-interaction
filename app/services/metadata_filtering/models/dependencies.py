from dataclasses import dataclass

from app.schemas.enums import SupportedAssetType
from app.services.metadata_filtering.models.outputs import LLM_NaturalLanguageCondition


@dataclass
class NormalizationAgentDeps:
    invalid_values: list[str]
    valid_values: list[str]


@dataclass
class NLConditionParsingDeps:
    nl_condition: LLM_NaturalLanguageCondition
    asset_type: SupportedAssetType
