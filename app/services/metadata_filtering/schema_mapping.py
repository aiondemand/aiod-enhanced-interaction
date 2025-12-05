from typing import Type
from app.schemas.asset_metadata.base_schemas import (
    AssetSpecific_AiExtractedMetadata,
    AssetSpecific_UserQueryParsedMetadata,
)
from app.schemas.asset_metadata.dataset_schema import (
    Dataset_AiExtractedMetadata,
    Dataset_UserQueryParsedMetadata,
)
from app.schemas.asset_metadata.educational_resource_schema import (
    EducationalResource_AiExtractedMetadata,
    EducationalResource_UserQueryParsedMetadata,
)
from app.schemas.asset_metadata.ml_model_schema import (
    MlModel_AiExtractedMetadata,
    MlModel_UserQueryParsedMetadata,
)
from app.schemas.asset_metadata.publication_schema import (
    Publication_AiExtractedMetadata,
    Publication_UserQueryParsedMetadata,
)
from app.schemas.enums import SupportedAssetType


ASSET_EXTRACTION_SCHEMA_MAPPING: dict[
    SupportedAssetType, Type[AssetSpecific_AiExtractedMetadata]
] = {
    SupportedAssetType.DATASETS: Dataset_AiExtractedMetadata,
    SupportedAssetType.ML_MODELS: MlModel_AiExtractedMetadata,
    SupportedAssetType.PUBLICATIONS: Publication_AiExtractedMetadata,
    SupportedAssetType.EDUCATIONAL_RESOURCES: EducationalResource_AiExtractedMetadata,
}


QUERY_PARSING_SCHEMA_MAPPING: dict[
    SupportedAssetType, Type[AssetSpecific_UserQueryParsedMetadata]
] = {
    SupportedAssetType.DATASETS: Dataset_UserQueryParsedMetadata,
    SupportedAssetType.ML_MODELS: MlModel_UserQueryParsedMetadata,
    SupportedAssetType.PUBLICATIONS: Publication_UserQueryParsedMetadata,
    SupportedAssetType.EDUCATIONAL_RESOURCES: EducationalResource_UserQueryParsedMetadata,
}
