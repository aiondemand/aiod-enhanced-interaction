from typing import Type
from app.schemas.asset_metadata.new_schemas.base_schemas import AssetSpecificMetadata
from app.schemas.asset_metadata.new_schemas.dataset_schema import Dataset_AiExtractedMetadata
from app.schemas.asset_metadata.new_schemas.educational_resource_schema import (
    EducationalResource_AiExtractedMetadata,
)
from app.schemas.asset_metadata.new_schemas.model_schema import Model_AiExtractedMetadata
from app.schemas.asset_metadata.new_schemas.publication_schema import (
    Publication_AiExtractedMetadata,
)
from app.schemas.enums import SupportedAssetType


METADATA_EXTRACTION_SCHEMA_MAPPING: dict[SupportedAssetType, Type[AssetSpecificMetadata]] = {
    SupportedAssetType.DATASETS: Dataset_AiExtractedMetadata,
    SupportedAssetType.ML_MODELS: Model_AiExtractedMetadata,
    SupportedAssetType.PUBLICATIONS: Publication_AiExtractedMetadata,
    SupportedAssetType.EDUCATIONAL_RESOURCES: EducationalResource_AiExtractedMetadata,
}
