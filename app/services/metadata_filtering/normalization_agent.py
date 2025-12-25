from functools import lru_cache
from pydantic_ai.settings import ModelSettings
from pydantic_ai import Agent, ModelRetry, ModelRetry, RunContext

from app.config import settings
from app.schemas.asset_metadata.base_schemas import (
    AssetSpecific_AiExtractedMetadata,
    AssetSpecific_UserQueryParsedMetadata,
)
from app.services.metadata_filtering.base import prepare_ollama_model
from app.services.metadata_filtering.models.dependencies import NormalizationAgentDeps
from app.services.metadata_filtering.models.outputs import LLM_NormalizedValue
from app.services.metadata_filtering.prompts.normalization_agent import NORMALIZATION_SYSTEM_PROMPT


class NormalizationAgent:
    def __init__(self) -> None:
        self.model = prepare_ollama_model()
        self.model_settings = ModelSettings(
            max_tokens=settings.OLLAMA.MAX_TOKENS,
        )

        self.agent = self.build_agent()

    def build_agent(self) -> Agent[NormalizationAgentDeps, list[LLM_NormalizedValue]]:
        return Agent(
            model=self.model,
            name="Normalization_Agent",
            system_prompt=NORMALIZATION_SYSTEM_PROMPT,
            output_type=self.transform_valid_values,
            output_retries=3,
            deps_type=NormalizationAgentDeps,
            model_settings=self.model_settings,
        )

    async def normalize_values(
        self,
        invalid_values: list[str],
        valid_values: list[str],
        pydantic_model: type[AssetSpecific_AiExtractedMetadata]
        | type[AssetSpecific_UserQueryParsedMetadata],
        field_name: str,
    ) -> list[str]:
        field_description = pydantic_model.get_described_fields()[field_name]
        user_prompt = self._build_user_prompt(
            invalid_values, valid_values, field_name, field_description
        )

        try:
            response = await self.agent.run(
                user_prompt, deps=NormalizationAgentDeps(invalid_values, valid_values)
            )
        except:
            return []

        normalized_values = [
            val.normalized_value for val in response.output if val.normalized_value is not None
        ]
        return list(set(normalized_values))

    def _build_user_prompt(
        self,
        invalid_values: list[str],
        valid_values: list[str],
        field_name: str,
        field_description: str,
    ) -> str:
        return (
            f"Normalize values for field: '{field_name}' ({field_description})\n\n"
            + f"Valid values: {valid_values}\n\n"
            + f"Values to normalize: {invalid_values}"
        )

    async def transform_valid_values(
        self, ctx: RunContext[NormalizationAgentDeps], normalized_values: list[LLM_NormalizedValue]
    ) -> list[LLM_NormalizedValue]:
        # Check whether normalized_values have been in fact normalized
        err_messages = []
        for it, norm_value in enumerate(normalized_values):
            original_value = ctx.deps.invalid_values[it]
            copied_original_value = norm_value.original_value
            normalized_value = norm_value.normalized_value

            if copied_original_value != original_value:
                err_messages.append(
                    f"Extracted original value '{copied_original_value}' doesn't match the actual original value '{original_value}' that was supposed to be normalized:"
                )
            elif normalized_value is not None and normalized_value not in ctx.deps.valid_values:
                err_messages.append(
                    f"Normalized value '{normalized_value}' corresponding to the original value '{copied_original_value}' that was supposed to be normalized is not one of the permitted values for this field."
                )

        if len(err_messages) > 0:
            raise ModelRetry("\n".join(err_messages))
        else:
            return normalized_values


@lru_cache()
def get_normalization_agent() -> NormalizationAgent | None:
    if settings.METADATA_FILTERING.ENABLED:
        return NormalizationAgent()
    else:
        return None
