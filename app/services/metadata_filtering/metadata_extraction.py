from typing import Awaitable, Callable, cast
from pydantic_ai.settings import ModelSettings
from pydantic_ai import Agent, ModelRetry, ModelRetry
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.schemas.asset_metadata.new_schemas.valid_values import field_valid_value_service
from app.schemas.asset_metadata.new_schemas.base_schemas import AssetSpecificMetadata
from app.schemas.asset_metadata.new_schemas.dataset_schema import Dataset_AiExtractedMetadata
from app.schemas.asset_metadata.new_schemas.educational_resource_schema import (
    EducationalResource_AiExtractedMetadata,
)
from app.schemas.asset_metadata.new_schemas.model_schema import MlModel_AiExtractedMetadata
from app.schemas.asset_metadata.new_schemas.publication_schema import (
    Publication_AiExtractedMetadata,
)
from app.schemas.asset_metadata.new_schemas.schema_mapping import METADATA_EXTRACTION_SCHEMA_MAPPING
from app.schemas.enums import SupportedAssetType
from app.config import settings
from app.services.metadata_filtering.prompts.metadata_extraction import (
    METADATA_EXTRACTION_SYSTEM_PROMPT,
)

# TODO Another agent that extracts the best match from enum for a arbitrary string or None/other


class MetadataExtractor:
    def __init__(self) -> None:
        # Ollama model
        self.model = OpenAIChatModel(
            model_name="qwen3:4b-instruct",
            provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
        )
        self.model_settings = ModelSettings(
            max_tokens=4_096,
        )
        self.agents = self.build_agents()

    def build_agents(self) -> dict[SupportedAssetType, Agent[None, AssetSpecificMetadata]]:
        agents: dict[SupportedAssetType, Agent[None, AssetSpecificMetadata]] = {}

        for asset_type in settings.AIOD.ASSET_TYPES_FOR_METADATA_EXTRACTION:
            agents[asset_type] = Agent(
                model=self.model,
                name=f"{asset_type.value.upper()}_MetadataExtractor_Agent",
                system_prompt=METADATA_EXTRACTION_SYSTEM_PROMPT,
                output_type=self._choose_output_function(asset_type),
                output_retries=2,
                model_settings=self.model_settings,
            )

        return agents

    def _choose_output_function(
        self, asset_type: SupportedAssetType
    ) -> Callable[[AssetSpecificMetadata], Awaitable[AssetSpecificMetadata]]:
        output_functions = {
            SupportedAssetType.DATASETS: self._extract_dataset_tool,
            SupportedAssetType.ML_MODELS: self._extract_ml_model_tool,
            SupportedAssetType.PUBLICATIONS: self._extract_publication_tool,
            SupportedAssetType.EDUCATIONAL_RESOURCES: self._extract_educational_resource_tool,
        }
        return output_functions[asset_type]  # type: ignore[return-value]

    async def extract_metadata(
        self, document: str, asset_type: SupportedAssetType
    ) -> AssetSpecificMetadata:
        try:
            run_output = await self.agents[asset_type].run(user_prompt=document)
        except Exception:
            # Empty model
            return METADATA_EXTRACTION_SCHEMA_MAPPING[asset_type]()

        return run_output.output

    # TODO LATER => We may want to check extracted values against enums...
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
        self, metadata: AssetSpecificMetadata, asset_type: SupportedAssetType
    ) -> AssetSpecificMetadata:
        fields_to_check = {
            field_name: field_value
            for field_name, field_value in metadata.model_dump().items()
            if field_value and field_valid_value_service.exists_values(asset_type, field_name)
        }

        # TODO LATER: We may want to use Pydantic validation errors instead
        errors = []
        for field_name, field_values in fields_to_check.items():
            valid_values = cast(
                list[str], field_valid_value_service.get_values(asset_type, field=field_name)
            )

            field_values = field_values if isinstance(field_values, list) else [field_values]
            invalid_values = [val for val in field_values if val not in valid_values]

            if len(invalid_values) > 0:
                errors.append(
                    f"The field '{field_name}' doesn't support the following values: {invalid_values}. The only allowed values for this field are: {valid_values}"
                )

        # TODO LATER: Narrow down the list of values...
        if len(errors) > 0:
            raise ModelRetry(
                f"The following validation errors found in {len(errors)} fields have been encountered:\n\n"
                + "\n".join(errors)
            )
        else:
            return metadata


metadata_extractor = MetadataExtractor()
