import json
from pathlib import Path

from app.schemas.enums import SupportedAssetType


class FieldValidValues:
    JSON_PATH = Path("app/data/field_valid_values.json")

    def __init__(self) -> None:
        if self.JSON_PATH.exists() is False:
            raise ValueError("JSON containing valid values for fields in the schemas doesn't exist")

        with open(self.JSON_PATH) as f:
            raw_data = json.load(f)

        base_fields = self._normalise_fields(raw_data.get("base", {}))
        base_fields = {field: list(dict.fromkeys(values)) for field, values in base_fields.items()}

        self.valid_values = {}
        for asset_type, fields in raw_data.items():
            if asset_type == "base" or not isinstance(fields, dict):
                continue

            specific_fields = self._normalise_fields(fields)
            merged_fields = self._merge_base_values(base_fields, specific_fields)
            self.valid_values[asset_type] = merged_fields

        self.base_values = base_fields

    @staticmethod
    def _normalise_fields(fields: dict) -> dict[str, list[str]]:
        return {
            field: [value.lower() if isinstance(value, str) else value for value in values]
            for field, values in fields.items()
            if isinstance(values, list)
        }

    @staticmethod
    def _merge_base_values(
        base_fields: dict[str, list[str]], specific_fields: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        merged: dict[str, list[str]] = {
            field: list(values) for field, values in base_fields.items()
        }
        for field, values in specific_fields.items():
            merged.setdefault(field, [])
            merged[field].extend(values)

        for field, values in merged.items():
            merged[field] = list(dict.fromkeys(values))

        return merged

    def get_values(self, asset_type: SupportedAssetType, field: str) -> list[str] | None:
        return self.valid_values.get(asset_type.value, {}).get(field, None)

    def exists_values(self, asset_type: SupportedAssetType, field: str) -> bool:
        return self.valid_values.get(asset_type.value, {}).get(field, None) is not None


field_valid_value_service = FieldValidValues()
