from pydantic_ai.settings import ModelSettings
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.schemas.asset_metadata.new_schemas.base_schemas import AssetSpecificMetadata
from app.schemas.asset_metadata.new_schemas.schema_mapping import METADATA_EXTRACTION_SCHEMA_MAPPING
from app.schemas.enums import SupportedAssetType
from app.config import settings
from app.services.metadata_filtering.prompts.metadata_extraction import (
    METADATA_EXTRACTION_SYSTEM_PROMPT,
)


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
                system_prompt=METADATA_EXTRACTION_SYSTEM_PROMPT,
                output_type=METADATA_EXTRACTION_SCHEMA_MAPPING[asset_type],
                output_retries=5,
                model_settings=self.model_settings,
            )

        return agents

    async def extract_metadata(
        self, document: str, asset_type: SupportedAssetType
    ) -> AssetSpecificMetadata:
        try:
            run_output = await self.agents[asset_type].run(user_prompt=document)
        except:
            # Empty model
            return METADATA_EXTRACTION_SCHEMA_MAPPING[asset_type]()

        return run_output.output


metadata_extractor = MetadataExtractor()
