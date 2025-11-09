from urllib.parse import urljoin

from pydantic_ai.models.openai import OpenAIChatModel

from app.config import settings
from app.schemas.asset_metadata.new_schemas.schema_mapping import METADATA_EXTRACTION_SCHEMA_MAPPING
from app.schemas.enums import SupportedAssetType
from app.services.metadata_filtering.models.outputs import (
    NaturalLanguageCondition_V2,
    StructedCondition_V2,
)
from app.services.metadata_filtering.nl_condition_parsing_agent import nl_condition_parsing_agent
from app.services.metadata_filtering.prompts.query_parsing_agent import QUERY_PARSING_SYSTEM_PROMPT

from functools import lru_cache
from urllib.parse import urljoin
from pydantic_ai.settings import ModelSettings
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.config import settings


class QueryParsingAgent:
    def __init__(self) -> None:
        # Ollama model
        ollama_url = urljoin(str(settings.OLLAMA.URI), "v1")
        self.model = OpenAIChatModel(
            model_name=settings.OLLAMA.MODEL_NAME,
            provider=OpenAIProvider(base_url=ollama_url),
        )
        self.model_settings = ModelSettings(
            max_tokens=settings.OLLAMA.MAX_TOKENS,
        )

        self.agents = self.build_agents()

    def build_agents(self) -> dict[SupportedAssetType, Agent[None, list[StructedCondition_V2]]]:
        agents = {}
        for asset_type in settings.AIOD.ASSET_TYPES_FOR_METADATA_EXTRACTION:
            described_fields = METADATA_EXTRACTION_SCHEMA_MAPPING[asset_type].get_described_fields()
            system_prompt = QUERY_PARSING_SYSTEM_PROMPT.format(described_fields=described_fields)

            agents[asset_type] = Agent(
                model=self.model,
                name=f"{asset_type.value.upper()}_UserQueryParsing_Agent",
                system_prompt=system_prompt,
                output_type=self._parse_nl_conditions,
                output_retries=3,
                deps_type=SupportedAssetType,
                model_settings=self.model_settings,
            )

        return agents

    async def extract_filters(
        self, user_query: str, asset_type: SupportedAssetType
    ) -> list[StructedCondition_V2]:
        try:
            user_prompt = f"Searching '{asset_type.value}' assets\n\nUser query:\n{user_query}"
            response = await self.agents[asset_type].run(user_prompt=user_prompt, deps=asset_type)
        except Exception:
            return []

        return response.output

    async def _parse_nl_conditions(
        self, ctx: RunContext[SupportedAssetType], nl_conditions: list[NaturalLanguageCondition_V2]
    ) -> list[StructedCondition_V2]:
        if nl_condition_parsing_agent is None:
            raise ValueError("Metadata Filtering is disabled")

        all_metadata_fields = list(
            METADATA_EXTRACTION_SCHEMA_MAPPING[ctx.deps].get_described_fields().keys()
        )

        # Check all the NLConditions are tied to existing metadata fields
        for nl_cond in nl_conditions:
            if nl_cond.field not in all_metadata_fields:
                raise ModelRetry(
                    f"Field '{nl_cond.field}' of one of the extracted natural language conditions does not exist. The only valid metadata fields are: {all_metadata_fields}"
                )

        # Go over individual NLConditions now
        structed_conditions = [
            await nl_condition_parsing_agent.build_filter(nl_cond, asset_type=ctx.deps)
            for nl_cond in nl_conditions
        ]
        return [cond for cond in structed_conditions if cond is not None]


@lru_cache()
def get_query_parsing_agent() -> QueryParsingAgent | None:
    if settings.METADATA_FILTERING.ENABLED:
        return QueryParsingAgent()
    else:
        return None


user_query_parsing_agent = get_query_parsing_agent()
