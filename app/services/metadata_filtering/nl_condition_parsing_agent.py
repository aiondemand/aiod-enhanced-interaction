from typing import cast

from pydantic import TypeAdapter, ValidationError

from app.config import settings
from app.services.metadata_filtering.base import prepare_ollama_model
from app.services.metadata_filtering.schema_mapping import QUERY_PARSING_SCHEMA_MAPPING
from app.services.metadata_filtering.field_valid_values import get_field_valid_values
from app.schemas.enums import SupportedAssetType
from app.services.metadata_filtering.models.dependencies import NLConditionParsingDeps
from app.services.metadata_filtering.models.outputs import (
    LLM_NaturalLanguageCondition,
    LLMStructedCondition,
)
from app.services.metadata_filtering.normalization_agent import get_normalization_agent
from app.services.metadata_filtering.prompts.nl_condition_parsing_agent import (
    NL_CONDITION_PARSING_SYSTEM_PROMPT,
)

from functools import lru_cache
from pydantic_ai.settings import ModelSettings
from pydantic_ai import Agent, ModelRetry, RunContext

from app.config import settings


class NLConditionParsingAgent:
    def __init__(self) -> None:
        self.model = prepare_ollama_model()
        self.model_settings = ModelSettings(
            max_tokens=settings.OLLAMA.MAX_TOKENS,
        )

        self.agent = self.build_agent()
        self.enforce_enums = settings.METADATA_FILTERING.ENFORCE_ENUMS

    def build_agent(self) -> Agent[NLConditionParsingDeps, LLMStructedCondition | None]:
        return Agent(
            model=self.model,
            name="NLConditionParsing_Agent",
            system_prompt=NL_CONDITION_PARSING_SYSTEM_PROMPT,
            output_type=self._build_and_validate_condition,
            output_retries=3,
            deps_type=NLConditionParsingDeps,
            model_settings=self.model_settings,
        )

    async def build_structed_condition(
        self, nl_condition: LLM_NaturalLanguageCondition, asset_type: SupportedAssetType
    ) -> LLMStructedCondition | None:
        try:
            user_prompt = self._build_user_prompt(nl_condition, asset_type)
            response = await self.agent.run(
                user_prompt=user_prompt,
                deps=NLConditionParsingDeps(nl_condition=nl_condition, asset_type=asset_type),
            )
        except:
            return None

        return response.output

    def _build_user_prompt(
        self, nl_condition: LLM_NaturalLanguageCondition, asset_type: SupportedAssetType
    ) -> str:
        pydantic_model = QUERY_PARSING_SCHEMA_MAPPING[asset_type]

        field_description = pydantic_model.get_described_fields()[nl_condition.field]
        inner_annotation = pydantic_model.get_inner_annotation(
            nl_condition.field, with_valid_values_enum=False
        )
        allowed_comparison_operators = pydantic_model.get_supported_comparison_operators(
            nl_condition.field
        )

        field_schema = TypeAdapter(inner_annotation).json_schema()
        field_schema.pop("title", None)
        field_schema.pop("description", None)

        metadata_field_string = f"Metadata field: '{nl_condition.field}'\nField description: {field_description}\nPermitted comparison operators: {allowed_comparison_operators}\nField schema: {field_schema}"
        condition_string = f"Natural language condition to analyze and extract expressions from: '{nl_condition.condition}'"
        return f"{metadata_field_string}\n\n{condition_string}"

    async def _build_and_validate_condition(
        self, ctx: RunContext[NLConditionParsingDeps], condition: LLMStructedCondition
    ) -> LLMStructedCondition | None:
        normalization_agent = get_normalization_agent()
        if normalization_agent is None:
            raise ValueError("Metadata Filtering is disabled")
        if condition.field != ctx.deps.nl_condition.field:
            raise ModelRetry(
                f"Incorrect metadata field. The condition works on top of '{ctx.deps.nl_condition.field}', not '{condition.field}'"
            )

        pydantic_model = QUERY_PARSING_SCHEMA_MAPPING[ctx.deps.asset_type]
        valid_enum_values: list[str] | None = (
            get_field_valid_values().get_values(ctx.deps.asset_type, field=condition.field)
            if self.enforce_enums
            else None
        )
        allowed_comparison_operators = pydantic_model.get_supported_comparison_operators(
            condition.field
        )

        valid_expressions = []
        for expr in condition.expressions:
            if expr.discard or expr.processed_value is None:
                continue

            # Validate the data type => strip of optional and list wrappers
            inner_annotation = pydantic_model.get_inner_annotation(
                condition.field, with_valid_values_enum=False
            )
            try:
                TypeAdapter(inner_annotation).validate_python(expr.processed_value)
            except ValidationError as e:
                raise ModelRetry(
                    f"Extracted processed_value '{expr.processed_value}' doesn't conform to the the value constraints imposed by the field definition.\n\n"
                    f"ValidationError: {e}"
                )

            # Check the comparison operator
            if expr.comparison_operator not in allowed_comparison_operators:
                raise ModelRetry(
                    f"Extracted comparison_operator '{expr.comparison_operator}' is not supported for the field '{condition.field}'. The only permitted comparison operators are: {allowed_comparison_operators}"
                )

            # Check against enum
            if valid_enum_values is not None and expr.processed_value not in valid_enum_values:
                normalized_values = await normalization_agent.normalize_values(
                    [cast(str, expr.processed_value)],
                    valid_enum_values,
                    pydantic_model,
                    condition.field,
                )
                # No valid value was mapped to the processed_value
                if len(normalized_values) == 0:
                    continue
                else:
                    expr.processed_value = normalized_values[0]

            # If the data type is correct and the value is one of the valid values (in the case of enum)
            # We store this expression
            valid_expressions.append(expr)

        if len(valid_expressions) > 0:
            return LLMStructedCondition(
                field=condition.field,
                logical_operator=condition.logical_operator,
                expressions=valid_expressions,
            )
        else:
            return None


@lru_cache()
def get_nl_condition_parsing_agent() -> NLConditionParsingAgent | None:
    if settings.METADATA_FILTERING.ENABLED:
        return NLConditionParsingAgent()
    else:
        return None
