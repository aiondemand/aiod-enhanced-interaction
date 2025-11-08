from typing import Any

from app.schemas.asset_metadata.new_schemas.base_schemas import AutomaticallyExtractedMetadata
from app.schemas.enums import SupportedAssetType
from app.services.inference.text_operations import ConvertJsonToString
from app.services.metadata_filtering.metadata_extraction_agent import metadata_extractor_agent


class MetadataExtractionWrapper:
    @classmethod
    def filter_out_empty_fields(cls, obj: dict) -> dict:
        def not_empty(val: Any) -> bool:
            if val is None:
                return False
            if isinstance(val, list) or isinstance(val, str):
                return len(val) > 0
            return True

        return {k: v for k, v in obj.items() if not_empty(v)}

    @classmethod
    async def extract_metadata(cls, obj: dict, asset_type: SupportedAssetType):
        if metadata_extractor_agent is None:
            raise ValueError("Metadata Filtering is disabled")

        # Deterministic extraction
        try:
            deterministic_fields = ["platform", "name", "date_published", "same_as"]
            kwargs = {field: obj.get(field, None) for field in deterministic_fields}
            deterministic_model = AutomaticallyExtractedMetadata(**kwargs)
        except:
            # Empty model
            deterministic_model = AutomaticallyExtractedMetadata()

        # Non-deterministic extraction (LLM-driven)
        obj_string = ConvertJsonToString.extract_relevant_info(obj, asset_type)
        non_deterministic_model = await metadata_extractor_agent.extract_metadata(
            obj_string, asset_type
        )

        return cls.filter_out_empty_fields(
            {**deterministic_model.model_dump(), **non_deterministic_model.model_dump()}
        )
