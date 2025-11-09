from app.models.filter import Filter
from app.schemas.enums import SupportedAssetType
from app.services.metadata_filtering.schema_mapping import SCHEMA_MAPPING


class MilvusTranslator:
    @classmethod
    def translate(cls, filters: list[Filter], asset_type: SupportedAssetType) -> str:
        def format_value(val: str | int | float) -> str:
            return f"'{val.lower()}'" if isinstance(val, str) else str(val)

        asset_schema = SCHEMA_MAPPING[asset_type]

        simple_expression_template = "({field} {op} {val})"
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
