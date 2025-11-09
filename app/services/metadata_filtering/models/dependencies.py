from dataclasses import dataclass

from app.schemas.enums import SupportedAssetType
from app.services.metadata_filtering.models.outputs import NaturalLanguageCondition_V2


@dataclass
class NormalizationAgentDeps:
    invalid_values: list[str]
    valid_values: list[str]


@dataclass
class NLConditionParsingDeps:
    nl_condition: NaturalLanguageCondition_V2
    asset_type: SupportedAssetType
