from functools import lru_cache
from typing import Any, Awaitable, Callable, cast
from pydantic_ai.settings import ModelSettings
from pydantic_ai import Agent

from app.services.metadata_filtering.base import prepare_ollama_model
from app.services.metadata_filtering.field_valid_values import field_valid_value_service
from app.schemas.asset_metadata.base_schemas import AssetSpecific_AiExtractedMetadata
from app.schemas.asset_metadata.dataset_schema import Dataset_AiExtractedMetadata
from app.schemas.asset_metadata.educational_resource_schema import (
    EducationalResource_AiExtractedMetadata,
)
from app.schemas.asset_metadata.ml_model_schema import MlModel_AiExtractedMetadata
from app.schemas.asset_metadata.publication_schema import (
    Publication_AiExtractedMetadata,
)
from app.services.metadata_filtering.schema_mapping import ASSET_EXTRACTION_SCHEMA_MAPPING
from app.schemas.enums import SupportedAssetType
from app.config import settings
from app.services.metadata_filtering.normalization_agent import normalization_agent
from app.services.metadata_filtering.prompts.metadata_extraction_agent import (
    METADATA_EXTRACTION_SYSTEM_PROMPT,
)
from app.schemas.asset_metadata.base_schemas import AutomaticallyExtractedMetadata
from app.schemas.enums import SupportedAssetType
from app.services.inference.text_operations import ConvertJsonToString


class MetadataExtractionWrapper:
    @classmethod
    def filter_out_empty_fields(cls, obj: dict) -> dict:
        def not_empty(val: Any) -> bool:
            if val is None:
                return False
            if isinstance(val, list) or isinstance(val, str):
                return len(val) > 0
            return True

        return {k: v for k, v in obj.items() if not_empty(v)}

    @classmethod
    async def extract_metadata(cls, obj: dict, asset_type: SupportedAssetType) -> dict:
        if metadata_extractor_agent is None:
            raise ValueError("Metadata Filtering is disabled")

        # Deterministic extraction
        try:
            deterministic_fields = ["platform", "name", "date_published", "same_as"]
            kwargs = {field: obj.get(field, None) for field in deterministic_fields}
            deterministic_model = AutomaticallyExtractedMetadata(**kwargs)
        except:
            # Empty model
            deterministic_model = AutomaticallyExtractedMetadata()

        # Non-deterministic extraction (LLM-driven)
        obj_string = ConvertJsonToString.extract_relevant_info(obj, asset_type)
        non_deterministic_model = await metadata_extractor_agent.extract_metadata(
            obj_string, asset_type
        )

        return cls.filter_out_empty_fields(
            {**deterministic_model.model_dump(), **non_deterministic_model.model_dump()}
        )


class MetadataExtractionAgent:
    def __init__(self) -> None:
        self.model = prepare_ollama_model()
        self.model_settings = ModelSettings(
            max_tokens=settings.OLLAMA.MAX_TOKENS,
        )

        self.enforce_enums = settings.METADATA_FILTERING.ENFORCE_ENUMS
        self.agents = self.build_agents()

    def build_agents(
        self,
    ) -> dict[SupportedAssetType, Agent[None, AssetSpecific_AiExtractedMetadata]]:
        agents = {}
        for asset_type in settings.AIOD.ASSET_TYPES_FOR_METADATA_EXTRACTION:
            agents[asset_type] = Agent(
                model=self.model,
                name=f"{asset_type.value.upper()}_MetadataExtractor_Agent",
                system_prompt=METADATA_EXTRACTION_SYSTEM_PROMPT,
                output_type=self._choose_output_function(asset_type),
                output_retries=3,
                model_settings=self.model_settings,
            )

        return agents

    def _choose_output_function(
        self, asset_type: SupportedAssetType
    ) -> Callable[
        [AssetSpecific_AiExtractedMetadata], Awaitable[AssetSpecific_AiExtractedMetadata]
    ]:
        output_functions = {
            SupportedAssetType.DATASETS: self._extract_dataset_tool,
            SupportedAssetType.ML_MODELS: self._extract_ml_model_tool,
            SupportedAssetType.PUBLICATIONS: self._extract_publication_tool,
            SupportedAssetType.EDUCATIONAL_RESOURCES: self._extract_educational_resource_tool,
        }
        return output_functions[asset_type]  # type: ignore[return-value]

    async def extract_metadata(
        self, document: str, asset_type: SupportedAssetType
    ) -> AssetSpecific_AiExtractedMetadata:
        try:
            user_prompt = f"Description ML {asset_type.value}:\n\n{document}"
            response = await self.agents[asset_type].run(user_prompt=user_prompt)
        except Exception:
            # Empty model
            return ASSET_EXTRACTION_SCHEMA_MAPPING[asset_type]()

        return response.output

    async def _extract_dataset_tool(
        self,
        metadata: Dataset_AiExtractedMetadata,
    ) -> Dataset_AiExtractedMetadata:
        """Extract metadata pertaining to a machine learning dataset"""
        # Check specific fields against AIoD taxonomies
        return await self.__extract_asset_metadata_tool(metadata, SupportedAssetType.DATASETS)  # type: ignore[return-value]

    async def _extract_ml_model_tool(
        self,
        metadata: MlModel_AiExtractedMetadata,
    ) -> MlModel_AiExtractedMetadata:
        """Extract metadata pertaining to a machine learning model"""
        return await self.__extract_asset_metadata_tool(metadata, SupportedAssetType.ML_MODELS)  # type: ignore[return-value]

    async def _extract_publication_tool(
        self,
        metadata: Publication_AiExtractedMetadata,
    ) -> Publication_AiExtractedMetadata:
        """Extract metadata pertaining to a machine learning publication"""
        return await self.__extract_asset_metadata_tool(metadata, SupportedAssetType.PUBLICATIONS)  # type: ignore[return-value]

    async def _extract_educational_resource_tool(
        self,
        metadata: EducationalResource_AiExtractedMetadata,
    ) -> EducationalResource_AiExtractedMetadata:
        """Extract metadata pertaining to a machine learning educational resource"""
        return await self.__extract_asset_metadata_tool(
            metadata,
            SupportedAssetType.EDUCATIONAL_RESOURCES,  # type: ignore[return-value]
        )

    async def __extract_asset_metadata_tool(
        self, metadata: AssetSpecific_AiExtractedMetadata, asset_type: SupportedAssetType
    ) -> AssetSpecific_AiExtractedMetadata:
        if self.enforce_enums:
            return await self.__validate_enums(metadata, asset_type)
        else:
            return metadata

    async def __validate_enums(
        self, metadata: AssetSpecific_AiExtractedMetadata, asset_type: SupportedAssetType
    ) -> AssetSpecific_AiExtractedMetadata:
        if normalization_agent is None:
            raise ValueError("Metadata Filtering is disabled")

        pydantic_model: type[AssetSpecific_AiExtractedMetadata] = metadata.__class__
        all_model_fields = metadata.model_dump()
        fields_to_check = {
            field_name: field_value
            for field_name, field_value in all_model_fields.items()
            if field_value and field_valid_value_service.exists_values(asset_type, field_name)
        }

        # Go over fields that we need to check against a list of valid values
        for field_name, field_values in fields_to_check.items():
            valid_values = cast(
                list[str], field_valid_value_service.get_values(asset_type, field=field_name)
            )
            # Preprocessing (wrap into a list, apply lowercase)
            is_field_a_list = isinstance(field_values, list)
            field_values = field_values if is_field_a_list else [field_values]
            field_values = [val.lower() for val in field_values]

            valid_extracted_values = [val for val in field_values if val in valid_values]
            invalid_extracted_values = [val for val in field_values if val not in valid_values]

            # Subagent for normalizing incorrect values
            normalized_extracted_values = []
            if len(invalid_extracted_values) > 0:
                normalized_extracted_values = await normalization_agent.normalize_values(
                    invalid_extracted_values, valid_values, pydantic_model, field_name
                )

            # Postprocessing (merging of values, list unwrapping if necessary)
            merged_extracted_values = valid_extracted_values + normalized_extracted_values
            if len(merged_extracted_values) == 0 and is_field_a_list is False:
                all_model_fields[field_name] = None
            else:
                all_model_fields[field_name] = (
                    merged_extracted_values if is_field_a_list else merged_extracted_values[0]
                )

        return pydantic_model(**all_model_fields)


@lru_cache()
def get_metadata_extraction_agent() -> MetadataExtractionAgent | None:
    if settings.METADATA_FILTERING.ENABLED:
        return MetadataExtractionAgent()
    else:
        return None


metadata_extractor_agent = get_metadata_extraction_agent()
