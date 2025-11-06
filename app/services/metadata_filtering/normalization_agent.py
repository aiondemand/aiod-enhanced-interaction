from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai.settings import ModelSettings
from pydantic_ai import Agent, ModelRetry, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.schemas.asset_metadata.new_schemas.base_schemas import AssetSpecificMetadata
from app.services.metadata_filtering.prompts.normalization_agent import NORMALIZATION_SYSTEM_PROMPT


@dataclass
class ModelInput:
    invalid_values: list[str]
    valid_values: list[str]


class NormalizedValue(BaseModel):
    original_value: str = Field(..., description="The input value before normalization.")
    normalized_value: str | None = Field(
        ..., description="The resulting value after applying normalization rules."
    )


class NormalizationAgent:
    def __init__(self) -> None:
        # Ollama model
        self.model = OpenAIChatModel(
            model_name="qwen3:4b-instruct",
            provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
        )
        self.model_settings = ModelSettings(
            max_tokens=1_024,
        )

        self.agent = self.build_agent()

    def build_agent(self) -> Agent[ModelInput, list[NormalizedValue]]:
        return Agent(
            model=self.model,
            name="Normalization_Agent",
            system_prompt=NORMALIZATION_SYSTEM_PROMPT,
            output_type=self.transform_valid_values,
            output_retries=2,
            deps_type=ModelInput,
            model_settings=self.model_settings,
        )

    async def normalize_values(
        self,
        invalid_values: list[str],
        valid_values: list[str],
        pydantic_model: type[AssetSpecificMetadata],
        field_name: str,
    ) -> list[str]:
        field_description = pydantic_model.model_fields[field_name].description
        field_description = field_description if field_description else ""

        user_prompt = self._build_user_prompt(
            invalid_values, valid_values, field_name, field_description
        )

        try:
            response = await self.agent.run(
                user_prompt, deps=ModelInput(invalid_values, valid_values)
            )
        except:
            return []

        values = set(
            [
                norm_value.normalized_value
                for norm_value in response.output
                if norm_value.normalized_value is not None
            ]
        )
        return list(values)

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
        self, ctx: RunContext[ModelInput], normalized_values: list[NormalizedValue]
    ) -> list[NormalizedValue]:
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


normalization_agent = NormalizationAgent()
