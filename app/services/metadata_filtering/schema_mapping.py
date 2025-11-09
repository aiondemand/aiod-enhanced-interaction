from typing import Type
from app.schemas.asset_metadata.base_schemas import AssetSpecificMetadata
from app.schemas.asset_metadata.dataset_schema import Dataset_AiExtractedMetadata
from app.schemas.asset_metadata.educational_resource_schema import (
    EducationalResource_AiExtractedMetadata,
)
from app.schemas.asset_metadata.model_schema import MlModel_AiExtractedMetadata
from app.schemas.asset_metadata.publication_schema import (
    Publication_AiExtractedMetadata,
)
from app.schemas.enums import SupportedAssetType


SCHEMA_MAPPING: dict[SupportedAssetType, Type[AssetSpecificMetadata]] = {
    SupportedAssetType.DATASETS: Dataset_AiExtractedMetadata,
    SupportedAssetType.ML_MODELS: MlModel_AiExtractedMetadata,
    SupportedAssetType.PUBLICATIONS: Publication_AiExtractedMetadata,
    SupportedAssetType.EDUCATIONAL_RESOURCES: EducationalResource_AiExtractedMetadata,
}
