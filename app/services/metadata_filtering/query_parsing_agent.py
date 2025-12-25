from functools import lru_cache
from pydantic_ai.settings import ModelSettings
from pydantic_ai import Agent, ModelRetry, RunContext

from app.models.filter import Filter
from app.config import settings
from app.services.metadata_filtering.base import prepare_ollama_model
from app.services.metadata_filtering.schema_mapping import QUERY_PARSING_SCHEMA_MAPPING
from app.schemas.enums import SupportedAssetType
from app.services.metadata_filtering.models.outputs import (
    LLM_NaturalLanguageCondition,
    LLMStructedCondition,
)
from app.services.metadata_filtering.nl_condition_parsing_agent import (
    get_nl_condition_parsing_agent,
)
from app.services.metadata_filtering.prompts.query_parsing_agent import QUERY_PARSING_SYSTEM_PROMPT
from app.config import settings


class QueryParsingWrapper:
    @classmethod
    async def parse_query(cls, user_query: str, asset_type: SupportedAssetType) -> dict:
        query_parsing_agent = get_query_parsing_agent()
        if query_parsing_agent is None:
            raise ValueError("Metadata Filtering is disabled")

        conditions = await query_parsing_agent.extract_conditions(user_query, asset_type)
        filters = [Filter.build_from_llm_condition(cond) for cond in conditions]

        return {
            "topic": user_query,
            "filter_str": cls.milvus_translate(filters, asset_type),
            "filters": filters,
        }

    @classmethod
    def milvus_translate(cls, filters: list[Filter], asset_type: SupportedAssetType) -> str:
        def format_value(val: str | int | float) -> str:
            return f"'{val.lower()}'" if isinstance(val, str) else str(val)

        asset_schema = QUERY_PARSING_SCHEMA_MAPPING[asset_type]

        simple_expression_template = "({field} {op} {val})"
        text_match_expression_template = "({op}TEXT_MATCH({field}, {val}))"
        list_expression_template = "({op}ARRAY_CONTAINS({field}, {val}))"
        list_fields_mask = asset_schema.get_list_fields_mask()

        condition_strings: list[str] = []
        for cond in filters:
            field = cond.field
            log_operator = cond.logical_operator

            str_expressions: list[str] = []
            for expr in cond.expressions:
                comp_operator = expr.comparison_operator
                val = expr.value

                # We work with a metadata field that has a list of values
                # We compare a value against a list of values
                if list_fields_mask[field]:
                    if comp_operator not in ["==", "!="]:
                        raise ValueError(
                            "We don't support any other comparison operators but a '==', '!=' for checking whether a value exist within a list of values (Milvus's 'ARRAY_CONTAINS' operation)."
                        )
                    str_expressions.append(
                        list_expression_template.format(
                            field=field,
                            op="" if comp_operator == "==" else "not ",
                            val=format_value(val),
                        )
                    )
                # 'name' field is special: The contents of this field are indexed and analyzed
                # We can use text matching akin to ElasticSearch
                elif field == "name":
                    if comp_operator not in ["==", "!="]:
                        raise ValueError(
                            "We don't support any other comparison operators but a '==', '!=' for performing a Milvus's 'TEXT_MATCH' operation."
                        )
                    str_expressions.append(
                        text_match_expression_template.format(
                            field=field,
                            op="" if comp_operator == "==" else "not ",
                            val=format_value(val),
                        )
                    )
                # Other non-list fields to compare a value against
                else:
                    str_expressions.append(
                        simple_expression_template.format(
                            field=field, op=comp_operator, val=format_value(val)
                        )
                    )
            condition_strings.append("(" + f" {log_operator.lower()} ".join(str_expressions) + ")")

        return " and ".join(condition_strings)


class QueryParsingAgent:
    def __init__(self) -> None:
        self.model = prepare_ollama_model()
        self.model_settings = ModelSettings(
            max_tokens=settings.OLLAMA.MAX_TOKENS,
        )

        self.agents = self.build_agents()

    def build_agents(self) -> dict[SupportedAssetType, Agent[None, list[LLMStructedCondition]]]:
        agents = {}
        for asset_type in settings.AIOD.ASSET_TYPES_FOR_METADATA_EXTRACTION:
            described_fields = QUERY_PARSING_SCHEMA_MAPPING[asset_type].get_described_fields()
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

    async def extract_conditions(
        self, user_query: str, asset_type: SupportedAssetType
    ) -> list[LLMStructedCondition]:
        try:
            user_prompt = f"Searching '{asset_type.value}' assets\n\nUser query:\n{user_query}"
            response = await self.agents[asset_type].run(user_prompt=user_prompt, deps=asset_type)
        except Exception:
            return []

        return response.output

    async def _parse_nl_conditions(
        self, ctx: RunContext[SupportedAssetType], nl_conditions: list[LLM_NaturalLanguageCondition]
    ) -> list[LLMStructedCondition]:
        nl_condition_parsing_agent = get_nl_condition_parsing_agent()
        if nl_condition_parsing_agent is None:
            raise ValueError("Metadata Filtering is disabled")

        all_metadata_fields = list(
            QUERY_PARSING_SCHEMA_MAPPING[ctx.deps].get_described_fields().keys()
        )

        # Check all the NLConditions are tied to existing metadata fields
        for nl_cond in nl_conditions:
            if nl_cond.field not in all_metadata_fields:
                raise ModelRetry(
                    f"Field '{nl_cond.field}' of one of the extracted natural language conditions does not exist. The only valid metadata fields are: {all_metadata_fields}"
                )

        # Go over individual NLConditions now
        structed_conditions = [
            await nl_condition_parsing_agent.build_structed_condition(nl_cond, asset_type=ctx.deps)
            for nl_cond in nl_conditions
        ]
        return [cond for cond in structed_conditions if cond is not None]


@lru_cache()
def get_query_parsing_agent() -> QueryParsingAgent | None:
    if settings.METADATA_FILTERING.ENABLED:
        return QueryParsingAgent()
    else:
        return None
