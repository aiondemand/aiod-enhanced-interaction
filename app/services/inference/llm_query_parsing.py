import json
import logging
import re
from ast import literal_eval
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal, Type, Optional
import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_ollama import ChatOllama
from ollama import Client as OllamaClient
from pydantic import BaseModel, Field, ValidationError, field_validator

from app.config import settings
from app.models.filter import Filter
from app.schemas.asset_metadata.base import BaseMetadataTemplate
from app.schemas.asset_metadata.operations import SchemaOperations
from app.schemas.enums import AssetType
from app.services.resilience import OllamaUnavailableException, with_retry_sync

# TODO refactor code to consolidate logic of LLM invocations into one service, one class
# That class then can be extended with the resilience wrapper
# First though, this file will likely be substantially changed because of issue:
# - https://github.com/aiondemand/aiod-enhanced-interaction/issues/19


# Classes pertaining to the first stage of user query parsing
class NaturalLanguageCondition(BaseModel):
    condition: str = Field(
        ...,
        description="Natural language condition corresponding to a particular metadata field we use for filtering. It may contain either only one value to be compared to metadata field, or multiple values if there's an OR logical operator in between those values",
    )
    field: str = Field(..., description="Name of the metadata field")

    # helper field used for better operator analysis. Even though the model doesn't assign operator correctly all the time
    # it's a way of forcing the model to focus on logical operators in between conditions
    # since the value of this attribute is unreliable we don't use it in the second stage
    operator: Literal["AND", "OR", "NONE"] = Field(
        ...,
        description="Logical operator used between multiple values pertaining to the same metadata field. If the condition describes only one value, set it to NONE instead.",
    )

    @field_validator("field", mode="before")
    @classmethod
    def validate_field(cls, v: str) -> str:
        return v.lower()

    @field_validator("operator", mode="before")
    @classmethod
    def validate_operator(cls, v: str) -> str:
        return v.upper()


# TODO rename
class UserQuery_Stage1_OutputSchema(BaseModel):
    """Extraction and parsing of conditions and a topic found within a user query"""

    topic: str = Field(
        ...,
        description="A topic or a main subject of the user query that user seeks for",
    )
    conditions: list[NaturalLanguageCondition] = Field(
        ..., description="Natural language conditions"
    )


class ValidValue(BaseModel):
    """Validation of a value against a list of permitted values"""

    validated_value: str = Field(
        ...,
        description="A value that has been validated against a list of permitted values for a particular field",
    )
    is_matched: bool = Field(
        ...,
        description="Whether the value has been matched against the list of permitted values or not",
    )


class PrepareLLM:
    @classmethod
    def setup_ollama_llm(cls) -> ChatOllama:
        ollama_uri = str(settings.OLLAMA.URI)
        model_name = settings.OLLAMA.MODEL_NAME

        try:
            client = OllamaClient(host=ollama_uri)
            client.pull(model_name)

            return ChatOllama(
                model=model_name,
                num_predict=settings.OLLAMA.NUM_PREDICT,
                num_ctx=settings.OLLAMA.NUM_CTX,
                base_url=ollama_uri,
            )
        except Exception as e:
            logging.error(e)
            logging.error("Ollama is unavailable. Application is being terminated now")
            os._exit(1)


class LlamaManualFunctionCalling:
    tool_prompt_template = """
        You have access to the following functions:

        Use the function '{function_name}' to '{function_description}':
        {function_schema}

        If you choose to call a function ONLY reply in the following format with no prefix or suffix:

        <function=example_function_name>{{\"example_name\": \"example_value\"}}</function>

        Reminder:
        - Function calls MUST follow the specified format, start with <function= and end with </function>
        - Required parameters MUST be specified
        - Only call one function at a time
        - Put the entire function call reply on one line
        - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
    """

    def __init__(
        self,
        llm: ChatOllama,
        pydantic_model: Type[BaseModel] | None,
        chat_prompt_no_system: ChatPromptTemplate,
        call_function: Callable[[RunnableSequence, dict], dict | None] | None = None,
    ) -> None:
        self.pydantic_model = pydantic_model
        self.call_function = call_function

        composite_prompt = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", "{system_prompt}"),
                ]
            )
            + chat_prompt_no_system
        )
        if pydantic_model is not None:
            composite_prompt = composite_prompt.partial(
                system_prompt=self.populate_tool_prompt(pydantic_model)
            )

        self.chain = (
            composite_prompt
            | llm
            | StrOutputParser()
            | RunnableLambda(self.convert_llm_string_output_to_tool)
        )

    def __call__(self, input: dict) -> dict | None:
        if self.call_function is not None:
            return self.call_function(self.chain, input)
        out = self.chain.invoke(input)

        if self.validate_output(out, self.pydantic_model):
            return out
        return None

    @classmethod
    def validate_output(cls, output: dict, pydantic_model: Type[BaseModel] | None) -> bool:
        if pydantic_model is not None:
            try:
                pydantic_model(**output)
            except ValidationError:
                return False
        return True

    @classmethod
    def transform_fewshot_examples(
        cls, pydantic_model: Type[BaseModel], examples: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        return [
            {
                "input": ex["input"],
                "output": f"<function={pydantic_model.__name__}>{json.dumps(ex['output'])}</function>",
            }
            for ex in examples
        ]

    def convert_llm_string_output_to_tool(self, response: str) -> dict | None:
        function_regex = r"<function=(\w+)>(.*?)</function>"
        match = re.search(function_regex, response)

        if match:
            _, args_string = match.groups()
            try:
                return literal_eval(args_string)
            except Exception:
                try:
                    return json.loads(args_string)
                except Exception:
                    logging.warning(f"Unable to convert LLM string to tool: '{response}'")

        return None

    @classmethod
    def populate_tool_prompt(cls, pydantic_model: Type[BaseModel]) -> str:
        tool_schema = cls.transform_simple_pydantic_schema_to_tool_schema(pydantic_model)

        return cls.tool_prompt_template.format(
            function_name=tool_schema["name"],
            function_description=tool_schema["description"],
            function_schema=json.dumps(tool_schema),
        )

    @classmethod
    def transform_simple_pydantic_schema_to_tool_schema(
        cls, pydantic_model: Type[BaseModel]
    ) -> dict:
        pydantic_schema = pydantic_model.model_json_schema()

        pydantic_schema.pop("type")
        pydantic_schema["name"] = pydantic_schema.pop("title")
        pydantic_schema["parameters"] = {
            "type": "object",
            "properties": pydantic_schema.pop("properties"),
            "required": pydantic_schema.pop("required"),
        }

        return pydantic_schema


class UserQueryParsing:
    DB_TRANSLATOR_FUNCS = {"milvus": "milvus_translator"}

    # TODO this current implementation can only work with one asset_type
    # few shot examples... We would need to dynamically assign them on LLM invocation
    # once we know which asset type a specific input is associated with
    _DEFAULT_PATH_TO_STAGE_1_FEWSHOTS = Path(
        "app/data/fewshot_examples/user_query_stage1/datasets.json"
    )

    @staticmethod
    @with_retry_sync(output_exception_cls=OllamaUnavailableException)
    def invoke_with_resilience(chain: RunnableSequence, input: dict) -> Any:
        return chain.invoke(input)

    @classmethod
    def get_db_translator_func(
        cls, technology: str
    ) -> Callable[[list[Filter], Type[BaseModel]], str]:
        return getattr(cls, cls.DB_TRANSLATOR_FUNCS[technology])

    def __init__(
        self,
        llm: BaseChatModel,
        db_to_translate: Literal["milvus"] = "milvus",
        stage_1_fewshot_examples_filepath: Path | None = None,
        stage_2_fewshot_examples_dirpath: Path | None = None,
    ) -> None:
        if not db_to_translate in ["milvus"]:
            raise ValueError(f"Invalid db_to_translate argument '{db_to_translate}'")
        self.translator_func = self.get_db_translator_func(db_to_translate)

        if stage_1_fewshot_examples_filepath is None:
            stage_1_fewshot_examples_filepath = self._DEFAULT_PATH_TO_STAGE_1_FEWSHOTS

        self.stage_pipe_1 = UserQueryParsingStages.init_stage_1(
            llm=llm, fewshot_examples_path=stage_1_fewshot_examples_filepath
        )
        self.stage_pipe_2 = UserQueryParsingStages.init_stage_2(
            llm=llm,
            fewshot_examples_dirpath=stage_2_fewshot_examples_dirpath,
            validation_step=UserQueryParsingStages.init_stage_3(llm=llm),
        )

    def __call__(self, user_query: str, asset_type: AssetType) -> dict:
        def expand_topic_query(
            topic_list: list[str], new_objects: list[dict], content_key: str
        ) -> list[str]:
            to_add = [
                item[content_key]
                for item in new_objects
                if (item.get("discard", False) and item.get(content_key, None) is not None)
            ]
            return topic_list + to_add

        if asset_type not in SchemaOperations.get_supported_asset_types():
            raise ValueError(f"Invalid asset_type argument '{asset_type}'")
        asset_schema = SchemaOperations.get_asset_schema(asset_type)

        stage_1_input = {"query": user_query, "asset_schema": asset_schema}
        out_stage_1 = self.stage_pipe_1(stage_1_input)
        if out_stage_1 is None:
            return {"topic": user_query, "filter_str": "", "conditions": []}

        topic_list = [out_stage_1["topic"]]
        topic_list = expand_topic_query(
            topic_list, out_stage_1["conditions"], content_key="condition"
        )

        parsed_conditions: list[dict] = []
        valid_conditions = [
            cond for cond in out_stage_1["conditions"] if cond.get("discard", False) is False
        ]
        for nl_cond in valid_conditions:
            input = {
                "condition": nl_cond["condition"],
                "field": nl_cond["field"],
                "asset_schema": asset_schema,
            }
            out_stage_2 = self.stage_pipe_2(input)
            if out_stage_2 is None:
                topic_list.append(nl_cond["condition"])
                continue

            topic_list = expand_topic_query(
                topic_list, out_stage_2["expressions"], content_key="raw_value"
            )

            valid_expressions = [
                expr
                for expr in out_stage_2["expressions"]
                if (
                    expr.get("discard", False) is False
                    and expr.get("processed_value", None) is not None
                )
            ]
            if len(valid_expressions) > 0:
                parsed_conditions.append(
                    {
                        "field": nl_cond["field"],
                        "logical_operator": out_stage_2["logical_operator"],
                        "expressions": [
                            {
                                "value": expr["processed_value"],
                                "comparison_operator": expr["comparison_operator"],
                            }
                            for expr in valid_expressions
                        ],
                    }
                )

        topic = " ".join(topic_list)
        filters = [Filter(**cond) for cond in parsed_conditions]
        filter_string = self.translator_func(filters, asset_schema)

        return {
            "topic": topic,
            "filter_str": filter_string,
            "filters": filters,
        }

    @classmethod
    def milvus_translator(
        cls, filters: list[Filter], asset_schema: Type[BaseMetadataTemplate]
    ) -> str:
        def format_value(val: str | int | float) -> str:
            return f"'{val.lower()}'" if isinstance(val, str) else str(val)

        simple_expression_template = "({field} {op} {val})"
        list_expression_template = "({op}ARRAY_CONTAINS({field}, {val}))"
        list_fields_mask = SchemaOperations.get_list_fields_mask(asset_schema)

        condition_strings: list[str] = []
        for cond in filters:
            field = cond.field
            log_operator = cond.logical_operator

            str_expressions: list[str] = []
            for expr in cond.expressions:
                comp_operator = expr.comparison_operator
                val = expr.value

                if list_fields_mask[field]:
                    if comp_operator not in ["==", "!="]:
                        raise ValueError(
                            "We don't support any other comparison operators but a '==', '!=' for checking whether values exist within the metadata field."
                        )
                    str_expressions.append(
                        list_expression_template.format(
                            field=field,
                            op="" if comp_operator == "==" else "not ",
                            val=format_value(val),
                        )
                    )
                else:
                    str_expressions.append(
                        simple_expression_template.format(
                            field=field, op=comp_operator, val=format_value(val)
                        )
                    )
            condition_strings.append("(" + f" {log_operator.lower()} ".join(str_expressions) + ")")

        return " and ".join(condition_strings)


class UserQueryParsingStages:
    task_instructions_stage1 = """
        Your task is to process a user query that may contain multiple natural language conditions and a general topic. Each condition may correspond to a specific metadata field and describes one or more values that should be compared against that field.
        These conditions are subsequently used to filter out unsatisfactory data in database. On the other hand, the topic is used in semantic search to find the most relevant documents to the thing user seeks for

        A simple schema below briefly describes all the metadata fields we use for filtering purposes:
        {model_schema}

        **Key Guidelines:**

        1. **Conditions and Metadata Fields:**
        - Each condition must clearly correspond to exactly one metadata field that we use for filtering purposes.
        - If a condition is associated with a metadata field not found in the defined schema, disregard that condition.
        - If a condition references multiple metadata fields (e.g., "published after 2020 or in image format"), split it into separate conditions with NONE operators, each tied to its respective metadata field.

        2. **Handling Multiple Values:**
        - If a condition references multiple values for a single metadata field (e.g., "dataset containing French or English"), include all the values in the natural language condition.
        - Specify the logical operator (AND/OR) that ties the values:
            - Use **AND** when the query requires all the values to match simultaneously.
            - Use **OR** when the query allows any of the values to match.

        3. **Natural Language Representation:**
        - Preserve the natural language form of the conditions. You're also allowed to modify them slightly to preserve their meaning once they're extracted from their context

        4. **Logical Operators for Conditions:**
        - Always include a logical operator (AND/OR) for conditions with multiple values.
        - For conditions with a single value, the logical operator is not required but can default to "NONE" for clarity.

        5. **Extract user query topic**
        - A topic is a concise, high-level description of the main subject of the query.
        - It should exclude specific filtering conditions but still capture the core concept or intent of the query.
        - If the query does not explicitly state a topic, infer it based on the overall context of the query.
    """

    task_instructions_stage2 = """
        Your task is to parse a single condition extracted from a user query and transform it into a structured format for further processing. The condition consists of one or more expressions combined with a logical operator.
        Validate whether each expression value can be unambiguously transformed into its processed valid counterpart compliant with the restrictions imposed on the metadata field. If transformation of the expression value is not clear and ambiguous, discard the expression instead.

        **Key Terminology:**
        1. **Expression**:
        - Represents a single comparison between a value and a metadata field.
        - Includes:
            - `raw_value`: The original value directly retrieved from the natural language condition.
            - `processed_value`: The transformed `raw_value`, converted to the appropriate data type and format based on the value and type restrictions imposed on the metadata field '{field_name}'. If `raw_value` cannot be unambiguously converted to a its valid counterpart complaint with metadata field constraints, set this field to string value "NONE".
            - `comparison_operator`: The operator used for comparison (e.g., >, <, ==, !=).
            - `discard`: A boolean value indicating whether the expression should be discarded (True if `raw_value` cannot be unambiguously transformed into a valid `processed_value`).

        2. **Condition**:
        - Consists of one or more expressions combined with a logical operator.
        - Includes:
            - `expressions`: A list of expressions (at least one).
            - `logical_operator`: The logical relationship between expressions (AND/OR). This operator only makes sense when there are multiple expressions. If there's only one expression, set this to AND

        **Input:**
        On input You will receive:
        - `condition**: The natural language condition extracted from the user query. This query should contain one or more expressions to be extracted.

        **Instructions**:
        1. Identify potentially all the expressions composing the condition. Each expression has its corresponding value and comparison_operator used to compare the value to metadata field for filtering purposes
        2. Make sure that you perform an unambiguous transformation of the raw value associated with each expression to its valid counterpart that is compliant with the restrictions imposed on the metadata field '{field_name}'. The metadata field description and the its value restrictions are the following:
            a) Description: {field_description}
            {field_valid_values}

            If the transformation of the raw value is not clear and ambiguous, discard the expression.
        3. Identify logical operator applied between expressions. There's only one operator (AND/OR) applied in between all expressions.
    """

    task_instructions_stage3 = """
        Your task is to determine whether a given `value` representing a field of an ML asset can be mapped to one of the values in a provided list of `permitted_values`. Use the following rules:

        1. **Input Details**:
        - **Field**: The specific field of an ML asset the value pertains to (e.g., "languages", "task_types", "license", ...).
        - **Value to assess**: The `value` that is subject of analysis and potential transformation.
        - **Permitted Values**: A list of valid values for this field.

        2. **Validation Rules**:
        - If the `value` can be unambiguously matched to one of the permitted values, return the matched value.
        - If the mapping is ambiguous or if the `value` cannot be matched, stick to the `value` as-is and mark it as unmatched.

        3. **Output Schema**:
        Return a JSON object with the following fields:
        ```json
        {{
            "validated_value": "string",
            "is_matched": true or false
        }}
    """

    @classmethod
    def _create_dynamic_stage2_schema(
        cls, field_name: str, asset_schema: Type[BaseMetadataTemplate]
    ) -> Type[BaseModel]:
        original_field = asset_schema.model_fields[field_name]
        new_field = SchemaOperations.get_inner_field_info(asset_schema, field_name)

        # combine descriptions
        if new_field.description != original_field.description and new_field.description != "":
            new_field.description = f"{original_field.description} {new_field.description}"
        else:
            new_field.description = original_field.description
        new_field.description = f"The processed value used to compare to metadata field '{field_name}', that adheres to the same constraints as the field: {new_field.description}."
        new_field.annotation = Optional[new_field.annotation]  # type: ignore[assignment]

        inner_class_dict = {
            "__annotations__": {
                "raw_value": str,
                "processed_value": new_field.annotation,
                "comparison_operator": Literal[
                    *asset_schema.get_supported_comparison_operators(field_name)
                ],
                "discard": bool,
            },
            # We have intentionally split the value into two separate fields, into raw_value and processed value as our model had trouble
            # properly processing the values immediately. By defining an explicit intermediate step, to write down the raw value before transforming it,
            # we have actually managed to improve the model performance
            "raw_value": Field(
                ...,
                description=f"The value used to compare to metadata field '{field_name}' in its raw state, extracted from the natural language condition",
            ),
            "processed_value": new_field,
            "comparison_operator": Field(
                ...,
                description=f"The comparison operator that determines how the value should be compared to the metadata field '{field_name}'.",
            ),
            "discard": Field(
                False,
                description="A boolean value indicating whether the expression should be discarded if 'raw_value' cannot be transformed into a valid 'processed_value'",
            ),
        }
        inner_class_dict.update(
            SchemaOperations.create_new_field_validators(
                asset_schema, orig_field_name=field_name, new_field_name="processed_value"
            )
        )
        expression_class = type(
            f"UserQuery_Stage2_Expression_{asset_schema.__name__}_{field_name}",
            (BaseModel,),
            inner_class_dict,
        )

        return type(
            f"UserQuery_Stage2_Filter_{asset_schema.__name__}_{field_name}",
            (BaseModel,),
            {
                "__annotations__": {
                    "expressions": list[expression_class],  # type: ignore[valid-type]
                    "logical_operator": Literal["AND", "OR"],
                },
                "__doc__": f"Parsing of one condition pertaining to metadata field '{field_name}'. Condition comprises one or more expressions used to for filtering purposes",
                "expressions": Field(
                    ...,
                    descriptions="List of expressions composing the entire condition. Each expression is associated with a particular value and a comparison operator used to compare the value to the metadata field.",
                ),
                "logical_operator": Field(
                    ...,
                    descriptions="The logical operator that performs logical operations (AND/OR) between multiple expressions. If there's only one expression set this value to 'AND'.",
                ),
            },
        )

    @classmethod
    def _call_function_stage_2(
        cls,
        chain: RunnableSequence,
        input: dict,
        fewshot_examples_dirpath: Path | None = None,
        validation_step: LlamaManualFunctionCalling | None = None,
    ) -> dict | None:
        metadata_field = input["field"]
        asset_schema = input["asset_schema"]
        dynamic_type = cls._create_dynamic_stage2_schema(metadata_field, asset_schema)

        chain_to_use = chain
        if fewshot_examples_dirpath is not None:
            examples_path = fewshot_examples_dirpath / f"{metadata_field}.json"
            if examples_path.exists():
                with examples_path.open() as f:
                    fewshot_examples = json.load(f)
                if len(fewshot_examples) > 0:
                    example_prompt = ChatPromptTemplate.from_messages(
                        [
                            ("user", "Condition: {input}"),
                            ("ai", "{output}"),
                        ]
                    )
                    fewshot_prompt = FewShotChatMessagePromptTemplate(
                        examples=LlamaManualFunctionCalling.transform_fewshot_examples(
                            dynamic_type, fewshot_examples
                        ),
                        example_prompt=example_prompt,
                    )
                    old_prompt: ChatPromptTemplate = chain.steps[0]
                    new_prompt = ChatPromptTemplate.from_messages(
                        [
                            *old_prompt.messages[:-1],
                            fewshot_prompt,
                            old_prompt.messages[-1],
                        ]
                    )
                    chain_to_use = RunnableSequence(new_prompt, *chain.steps[1:])

        field_valid_values = ""
        curr_validation_step = None
        permitted_values: list[str] = []

        if SchemaOperations.exists_list_of_valid_values(asset_schema, metadata_field):
            permitted_values = SchemaOperations.get_list_of_valid_values(
                asset_schema, metadata_field
            )
            field_valid_values = f"b) List of the only permitted values: {permitted_values}"
            curr_validation_step = validation_step

        input_variables = {
            "query": input["condition"],
            "field_name": metadata_field,
            "field_description": asset_schema.model_fields[metadata_field].description,
            "field_valid_values": field_valid_values,
            "system_prompt": LlamaManualFunctionCalling.populate_tool_prompt(dynamic_type),
        }
        return cls._try_invoke_stage_2(
            chain_to_use,
            input_variables,
            dynamic_type,
            validation_step=curr_validation_step,
            validation_step_permitted_values=permitted_values,
        )

    @classmethod
    def prepare_simplified_model_schema_stage_1(cls, asset_schema: Type[BaseModel]) -> str:
        metadata_field_info = [
            {
                "name": name,
                "description": field.description,
                "type": SchemaOperations.translate_primitive_type_to_str(
                    SchemaOperations.get_inner_most_primitive_type(field.annotation)
                ),
            }
            for name, field in asset_schema.model_fields.items()
        ]
        return json.dumps(metadata_field_info)

    @classmethod
    def _try_invoke_stage_1(
        cls, chain: RunnableSequence, input: dict, num_retry_attempts: int = 5
    ) -> dict | None:
        def exists_conditions_list_in_wrapper_dict(obj: dict) -> bool:
            return obj.get("conditions", None) is not None and isinstance(obj["conditions"], list)

        def is_valid_wrapper_class(obj: dict, valid_field_names: list[str]) -> bool:
            try:
                UserQuery_Stage1_OutputSchema(**obj)

                invalid_fields = [
                    cond["field"]
                    for cond in obj["conditions"]
                    if cond["field"] not in valid_field_names
                ]
                if len(invalid_fields) == 0:
                    return True
            except ValidationError:
                pass
            return False

        def is_valid_condition_class(obj: Any, valid_field_names: list[str]) -> bool:
            if isinstance(obj, dict) is False:
                return False
            try:
                NaturalLanguageCondition(**obj)

                if obj["field"] in valid_field_names:
                    return True
            except ValidationError:
                pass
            return False

        best_llm_response = None
        max_valid_conditions_count = 0

        asset_schema = input["asset_schema"]
        simple_model_schema = cls.prepare_simplified_model_schema_stage_1(asset_schema)

        for _ in range(num_retry_attempts):
            output = UserQueryParsing.invoke_with_resilience(
                chain,
                input={
                    "query": input["query"],
                    "model_schema": simple_model_schema,
                },
            )
            if output is None:
                continue
            valid_field_names = list(asset_schema.model_fields.keys())
            if is_valid_wrapper_class(output, valid_field_names):
                return UserQuery_Stage1_OutputSchema(**output).model_dump()

            # The LLM output is invalid, now we will identify
            # which conditions are incorrect and how many are valid
            if exists_conditions_list_in_wrapper_dict(output) is False:
                continue
            valid_conditions_count = 0
            for i in range(len(output["conditions"])):
                if is_valid_condition_class(output["conditions"][i], valid_field_names):
                    valid_conditions_count += 1
                elif isinstance(output["conditions"][i], dict):
                    output["conditions"][i]["discard"] = True
                else:
                    output["conditions"][i] = {"discard": True}

            # we compare current LLM output to potentially previous LLm outputs
            # and identify the best LLM response (containing the most valid conditions)
            if valid_conditions_count > max_valid_conditions_count:
                # check whether the entire object is correct once we get
                # rid of invalid conditions
                helper_object = deepcopy(output)
                helper_object["conditions"] = [
                    cond for cond in output["conditions"] if cond.get("discard", False) is False
                ]
                if is_valid_wrapper_class(helper_object, valid_field_names):
                    best_llm_response = UserQuery_Stage1_OutputSchema(**helper_object).model_dump()
                    max_valid_conditions_count = valid_conditions_count

        return best_llm_response

    @classmethod
    def _try_invoke_stage_2(
        cls,
        chain: RunnableSequence,
        input: dict,
        wrapper_schema: Type[BaseModel],
        validation_step: LlamaManualFunctionCalling | None,
        validation_step_permitted_values: list[str] = [],
        num_retry_attempts: int = 5,
    ) -> dict | None:
        def exists_expressions_list_in_wrapper_dict(obj: dict) -> bool:
            return obj.get("expressions", None) is not None and isinstance(obj["expressions"], list)

        def is_valid_wrapper_class(obj: dict) -> bool:
            try:
                wrapper_schema(**obj)
                return True
            except ValidationError:
                return False

        def is_valid_expression_class(obj: dict) -> bool:
            try:
                expression_schema(**obj)
                return True
            except ValidationError:
                return False

        best_llm_response = None
        max_valid_expressions_count = 0
        expression_schema = SchemaOperations.strip_list_type(
            wrapper_schema.__annotations__["expressions"]
        )

        for _ in range(num_retry_attempts):
            output = UserQueryParsing.invoke_with_resilience(chain, input)
            if output is None:
                continue
            if is_valid_wrapper_class(output):
                return wrapper_schema(**output).model_dump()

            # The LLM output is invalid, now we will identify
            # which expressions are incorrect and how many are valid
            if exists_expressions_list_in_wrapper_dict(output) is False:
                continue
            valid_expressions_count = 0
            for i in range(len(output["expressions"])):
                if is_valid_expression_class(output["expressions"][i]):
                    valid_expressions_count += 1
                elif isinstance(output["expressions"][i], dict):
                    output["expressions"][i]["discard"] = True

                    # Apply validation step to compare extracted value against the list of valid/permitted values
                    if validation_step is not None:
                        validated_output = validation_step(
                            input={
                                "field": input["field_name"],
                                "value": output["expressions"][i]["processed_value"],
                                "permitted_values": validation_step_permitted_values,
                            }
                        )
                        if validated_output is not None:
                            output["expressions"][i]["processed_value"] = validated_output[
                                "validated_value"
                            ]
                            if is_valid_expression_class(output["expressions"][i]):
                                valid_expressions_count += 1
                                output["expressions"][i]["discard"] = False
                else:
                    output["expressions"][i] = {"discard": True}

            # we compare current LLM output to potentially previous LLm outputs
            # and identify the best LLM response (containing the most valid expressions)
            if valid_expressions_count > max_valid_expressions_count:
                # check whether the entire object is correct once we get
                # rid of invalid expressions
                helper_object = deepcopy(output)
                helper_object["expressions"] = [
                    expr for expr in output["expressions"] if expr.get("discard", False) is False
                ]
                if is_valid_wrapper_class(helper_object):
                    best_llm_response = wrapper_schema(**helper_object).model_dump()
                    max_valid_expressions_count = valid_expressions_count

        return best_llm_response

    @classmethod
    def _try_invoke_stage_3(
        cls, chain: RunnableSequence, input: dict, num_retry_attempts: int = 5
    ) -> dict | None:
        for _ in range(num_retry_attempts):
            output = UserQueryParsing.invoke_with_resilience(chain, input)
            if output is None:
                continue
            try:
                return ValidValue(**output).model_dump()
            except ValidationError:
                pass

        return None

    @classmethod
    def init_stage_1(
        cls,
        llm: BaseChatModel,
        fewshot_examples_path: Path | None = None,
    ) -> LlamaManualFunctionCalling:
        pydantic_model = UserQuery_Stage1_OutputSchema

        task_instructions = HumanMessagePromptTemplate.from_template(
            cls.task_instructions_stage1,
        )
        fewshot_prompt = ("user", "")
        if fewshot_examples_path is not None and fewshot_examples_path.exists():
            with fewshot_examples_path.open() as f:
                fewshot_examples = json.load(f)
            if len(fewshot_examples) > 0:
                example_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("user", "User Query: {input}"),
                        ("ai", "{output}"),
                    ]
                )
                fewshot_prompt = FewShotChatMessagePromptTemplate(
                    examples=LlamaManualFunctionCalling.transform_fewshot_examples(
                        pydantic_model, fewshot_examples
                    ),
                    example_prompt=example_prompt,
                )

        chat_prompt_no_system = ChatPromptTemplate.from_messages(
            [
                task_instructions,
                fewshot_prompt,
                ("user", "User Query: {query}"),
            ]
        )
        return LlamaManualFunctionCalling(
            llm,
            pydantic_model=pydantic_model,
            chat_prompt_no_system=chat_prompt_no_system,
            call_function=cls._try_invoke_stage_1,
        )

    @classmethod
    def init_stage_2(
        cls,
        llm: BaseChatModel,
        fewshot_examples_dirpath: Path | None = None,
        validation_step: LlamaManualFunctionCalling | None = None,
    ) -> LlamaManualFunctionCalling:
        chat_prompt_no_system = ChatPromptTemplate.from_messages(
            [
                ("user", cls.task_instructions_stage2),
                ("user", "Condition: {query}"),
            ]
        )
        return LlamaManualFunctionCalling(
            llm,
            pydantic_model=None,
            chat_prompt_no_system=chat_prompt_no_system,
            call_function=partial(
                cls._call_function_stage_2,
                fewshot_examples_dirpath=fewshot_examples_dirpath,
                validation_step=validation_step,
            ),
        )

    @classmethod
    def init_stage_3(cls, llm: BaseChatModel) -> LlamaManualFunctionCalling:
        chat_prompt_no_system = ChatPromptTemplate.from_messages(
            [
                ("user", cls.task_instructions_stage3),
                ("user", "**Field**: {field}"),
                ("user", "**Permitted values**: {permitted_values}"),
                ("user", "**Value to assess**: {value}"),
            ]
        )
        return LlamaManualFunctionCalling(
            llm,
            pydantic_model=ValidValue,
            chat_prompt_no_system=chat_prompt_no_system,
            call_function=partial(cls._try_invoke_stage_3, num_retry_attempts=2),
        )
