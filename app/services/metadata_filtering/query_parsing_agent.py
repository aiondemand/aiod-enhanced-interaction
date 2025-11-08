from typing import Literal, TypeAlias
from urllib.parse import urljoin

from pydantic import BaseModel, Field
from pydantic_ai.models.openai import OpenAIChatModel

from app.config import settings
from app.schemas.asset_metadata.new_schemas.schema_mapping import METADATA_EXTRACTION_SCHEMA_MAPPING
from app.schemas.enums import SupportedAssetType
from app.services.metadata_filtering.prompts.query_parsing_agent import (
    QUERY_PARSING_STAGE_1_SYSTEM_PROMPT,
    QUERY_PARSING_STAGE_2_SYSTEM_PROMPT,
)

from functools import lru_cache
from urllib.parse import urljoin
from pydantic import BaseModel, Field
from pydantic_ai.settings import ModelSettings
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.config import settings


# TODO use the same approach of parsing the conditions for now
# Same system prompts
# Same Pydantic models

# TODO WHAT TO CHANGE
# Technology: Langchain -> Pydantic AI
# Schema
# No few-shot examples for now


PrimitiveTypes: TypeAlias = int | float | str | bool


class Expression_V2(BaseModel):
    """An Expression represents a single comparison between a value and a metadata field"""

    raw_value: str = Field(
        ..., description="Raw value directly extracted from the natural language condition"
    )
    processed_value: PrimitiveTypes | None = Field(
        ...,
        description=(
            "Transformed original value adhering to value constraints of the specific metadata field. "
            "Value to be compared to the metadata field. "
            "If the original value cannot be unambiguosly mapped to one of the valid values, this field is set to None."
        ),
    )
    comparison_operator: Literal["<", ">", "<=", ">=", "==", "!="] = Field(
        ...,
        description="Comparison operator to be used for comparing the value to the metadata field",
    )
    discard: bool = Field(
        ...,
        description="If the value found within the natural language condition cannot be unambiguously mapped to its valid counterpart value, set this to True",
    )


class StructedCondition_V2(BaseModel):
    """A Condition consists of one or more expressions joined with a logical operator"""

    field: str = Field(..., description="Name of the metadata field to filter by")
    logical_operator: Literal["AND", "OR"] = Field(
        ..., description="Allowed logical operator to be used for combining multiple expressions"
    )
    expressions: list[Expression_V2] = Field(
        ...,
        description="List of expressions associated with their respective values and comparison operators to be used for filtering",
    )


class NaturalLanguageCondition_V2(BaseModel):
    """Condition in its natural language form extracted from user query"""

    condition: str = Field(
        ...,
        description=(
            "Natural language condition corresponding to a particular metadata field we use for filtering. "
            "It may contain either only one value to be compared to metadata field, "
            "or multiple values if there's an OR logical operator in between those values"
        ),
    )
    field: str = Field(..., description="Name of the metadata field")
    operator: Literal["AND", "OR", "NONE"] = Field(
        ...,
        description=(
            "Logical operator used between multiple values pertaining to the same metadata field. "
            "If the condition describes only one value, set it to NONE instead."
        ),
    )


class UserQueryParsingAgent:
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
            system_prompt = QUERY_PARSING_STAGE_1_SYSTEM_PROMPT.format(
                described_fields=described_fields
            )

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

    # TODO discarding logic may bubble up into a NL conditions?
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


class NLConditionParsingAgent:
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

        self.agent = self.build_agent()

    def build_agent(self) -> Agent[SupportedAssetType, StructedCondition_V2]:
        return Agent(
            model=self.model,
            name="NLConditionParsing_Agent",
            system_prompt=QUERY_PARSING_STAGE_2_SYSTEM_PROMPT,
            output_type=self._build_and_validate_filter,
            output_retries=3,
            deps_type=SupportedAssetType,
            model_settings=self.model_settings,
        )

    async def build_filter(
        self, nl_condition: NaturalLanguageCondition_V2, asset_type: SupportedAssetType
    ) -> StructedCondition_V2 | None:
        try:
            user_prompt = self._build_user_prompt(nl_condition, asset_type)
            response = await self.agent.run(user_prompt=user_prompt, deps=asset_type)
        except:
            return None

        return response.output

    def _build_user_prompt(
        self, nl_condition: NaturalLanguageCondition_V2, asset_type: SupportedAssetType
    ) -> str:
        pydantic_model = METADATA_EXTRACTION_SCHEMA_MAPPING[asset_type]
        schema = pydantic_model.model_json_schema()
        field_schema = dict(schema["properties"][nl_condition.field])
        field_schema.pop("title", None)

        field_description = pydantic_model.get_described_fields()[nl_condition.field]

        metadata_field_string = f"Metadata field: '{nl_condition.field}'\nField description:{field_description}\n\nField schema:{field_schema}"
        condition_string = f"Natural language condition: {nl_condition.condition}"

        return f"{metadata_field_string}\n\n{condition_string}"

    async def _build_and_validate_filter(
        self, filter: StructedCondition_V2
    ) -> StructedCondition_V2:
        # NEW SECOND STAGE
        # Input NLCondition
        # Output: Output function
        # Input: Filter
        # In function itself: checking validity...
        # Check data type, value constraints imposed by the field
        # enum -> normalized agent
        # Output: Filter

        pass


@lru_cache()
def get_user_query_parsing_agent() -> UserQueryParsingAgent | None:
    if settings.METADATA_FILTERING.ENABLED:
        return UserQueryParsingAgent()
    else:
        return None


@lru_cache()
def get_nl_condition_parsing_agent() -> NLConditionParsingAgent | None:
    if settings.METADATA_FILTERING.ENABLED:
        return NLConditionParsingAgent()
    else:
        return None


user_query_parsing_agent = get_user_query_parsing_agent()
nl_condition_parsing_agent = get_nl_condition_parsing_agent()
