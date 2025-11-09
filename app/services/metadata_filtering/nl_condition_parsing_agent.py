from types import UnionType
from typing import Type, Union, cast, get_args, get_origin
from urllib.parse import urljoin

from pydantic import TypeAdapter, ValidationError
from pydantic_ai.models.openai import OpenAIChatModel

from app.config import settings
from app.schemas.asset_metadata.new_schemas.schema_mapping import METADATA_EXTRACTION_SCHEMA_MAPPING
from app.schemas.asset_metadata.new_schemas.valid_values import field_valid_value_service
from app.schemas.enums import SupportedAssetType
from app.services.metadata_filtering.models.dependencies import NLConditionParsingDeps
from app.services.metadata_filtering.models.outputs import (
    NaturalLanguageCondition_V2,
    StructedCondition_V2,
)
from app.services.metadata_filtering.normalization_agent import normalization_agent
from app.services.metadata_filtering.prompts.nl_condition_parsing_agent import (
    NL_CONDITION_PARSING_SYSTEM_PROMPT,
)

from functools import lru_cache
from urllib.parse import urljoin
from pydantic_ai.settings import ModelSettings
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.config import settings


# TODO put somewhere else
class AnnotationOperations:
    @classmethod
    def strip_optional_and_list_types(cls, annotation: Type) -> Type:
        """Strip annotation of the form Optional[List[TYPE]] to TYPE"""
        return cls.strip_list_type(cls.strip_optional_type(annotation))

    @classmethod
    def is_optional_type(cls, annotation: Type) -> bool:
        if get_origin(annotation) is Union or get_origin(annotation) is UnionType:
            return type(None) in get_args(annotation)
        return False

    @classmethod
    def is_list_type(cls, annotation: Type) -> bool:
        return get_origin(annotation) is list

    @classmethod
    def strip_optional_type(cls, annotation: Type) -> Type:
        if cls.is_optional_type(annotation):
            return next(arg for arg in get_args(annotation) if arg is not type(None))
        return annotation

    @classmethod
    def strip_list_type(cls, annotation: Type) -> Type:
        if cls.is_list_type(annotation):
            return get_args(annotation)[0]
        return annotation


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

    def build_agent(self) -> Agent[NLConditionParsingDeps, StructedCondition_V2 | None]:
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
        self, nl_condition: NaturalLanguageCondition_V2, asset_type: SupportedAssetType
    ) -> StructedCondition_V2 | None:
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
        self, nl_condition: NaturalLanguageCondition_V2, asset_type: SupportedAssetType
    ) -> str:
        pydantic_model = METADATA_EXTRACTION_SCHEMA_MAPPING[asset_type]
        inner_annotation = self._get_field_inner_annotation(
            asset_type=asset_type, field_name=nl_condition.field
        )

        field_schema = TypeAdapter(inner_annotation).json_schema()
        field_schema.pop("title", None)
        field_schema.pop("description", None)
        field_description = pydantic_model.get_described_fields()[nl_condition.field]

        metadata_field_string = f"Metadata field: '{nl_condition.field}'\nField description: {field_description}\n\nField schema: {field_schema}"
        condition_string = f"Natural language condition to analyze and extract expressions from: '{nl_condition.condition}'"

        return f"{metadata_field_string}\n\n{condition_string}"

    def _get_field_inner_annotation(self, asset_type: SupportedAssetType, field_name: str) -> type:
        pydantic_model = METADATA_EXTRACTION_SCHEMA_MAPPING[asset_type]

        field_info = pydantic_model.model_fields[field_name]
        annotation = field_info.annotation
        if annotation is None:
            raise ValueError(
                f"Annotation for the field '{field_name}' for the asset '{asset_type}' doesn't exist. Fix the asset schema."
            )
        return AnnotationOperations.strip_optional_and_list_types(annotation)

    async def _build_and_validate_condition(
        self, ctx: RunContext[NLConditionParsingDeps], condition: StructedCondition_V2
    ) -> StructedCondition_V2 | None:
        if normalization_agent is None:
            raise ValueError("Metadata Filtering is disabled")
        if condition.field != ctx.deps.nl_condition.field:
            raise ModelRetry(
                f"Incorrect metadata field. The condition works on top of '{ctx.deps.nl_condition.field}', not '{condition.field}'"
            )

        valid_enum_values: list[str] | None = field_valid_value_service.get_values(
            ctx.deps.asset_type, field=condition.field
        )

        valid_expressions = []
        for expr in condition.expressions:
            if expr.discard or expr.processed_value is None:
                continue

            # Validate the data type => strip of optional and list wrappers
            inner_annotation = self._get_field_inner_annotation(
                asset_type=ctx.deps.asset_type, field_name=condition.field
            )
            try:
                TypeAdapter(inner_annotation).validate_python(expr.processed_value)
            except ValidationError as e:
                raise ModelRetry(
                    f"Extracted processed_value '{expr.processed_value}' doesn't conform to the the value constraints imposed by the field definition.\n\n"
                    f"ValidationError: {e}"
                )

            # Check against enum
            if valid_enum_values is not None and expr.processed_value not in valid_enum_values:
                normalized_values = await normalization_agent.normalize_values(
                    [cast(str, expr.processed_value)],
                    valid_enum_values,
                    ctx.deps.asset_type,
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
            return StructedCondition_V2(
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


nl_condition_parsing_agent = get_nl_condition_parsing_agent()
