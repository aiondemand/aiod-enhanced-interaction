import json
import re

from app.schemas.enums import SupportedAssetType


class ConvertJsonToString:
    orig_flat_fields = [
        "platform",
        "name",
        "date_published",
        "type",
    ]
    orig_array_fields = [
        "keyword",
        "alternate_name",
        "application_area",
        "industrial_sector",
        "research_area",
        "scientific_domain",
        "badge",
    ]
    orig_distrib_fields = ["distribution", "media"]

    educational_flat_fields = ["time_required", "pace"]
    educational_array_fields = [
        "access_mode",
        "educational_level",
        "in_language",
        "prerequisite",
        "target_audience",
    ]
    experiment_flat_fields = [
        "experimental_workflow",
        "execution_settings",
        "reproducibility_explanation",
    ]
    service_flat_fields = ["slogan", "terms_of_service"]

    @classmethod
    def stringify(cls, data: dict) -> str:
        return json.dumps(data)

    # "BASIC INFO"
    @classmethod
    def extract_very_basic_info(
        cls, data: dict, stringify: bool = False, include_id: bool = False
    ) -> str:
        simple_data = cls._extract_very_basic_fields(data)
        if stringify:
            return json.dumps(simple_data)

        description = simple_data.get("description", None)
        keywords = simple_data.get("keyword", None)

        string = f"Dataset ID: {simple_data['id']}\n" if include_id else ""
        string += f"Platform: {simple_data['platform']}\nAsset name: {simple_data['name']}"
        if description is not None:
            string += f"\nDescription: {description}"
        if keywords is not None:
            key_string = " | ".join(keywords)
            string += f"\nKeywords: {key_string}"

        return string

    # "RELEVANT INFO"
    @classmethod
    def extract_relevant_info(
        cls, data: dict, asset_type: SupportedAssetType, stringify: bool = False
    ) -> str:
        simple_data = cls._extract_relevant_fields(data, asset_type)
        if stringify:
            return json.dumps(simple_data)

        string: str = ""
        flat_fields: list[str] = []
        array_fields: list[str] = []
        distrib_fields: list[str] = []

        for k, v in simple_data.items():
            if k in cls.orig_distrib_fields:
                distrib_fields.append(k)
            elif isinstance(v, str) or isinstance(v, int) or isinstance(v, float):
                flat_fields.append(k)
            elif isinstance(v, list):
                array_fields.append(k)
            else:
                raise ValueError("Unknown field to deal with")

        for flat_field in flat_fields:
            string += f"{flat_field}: {simple_data[flat_field]}\n"
        string += "\n"

        for array_field in array_fields:
            string += f"{array_field}: {', '.join(simple_data[array_field])}\n"
        string += "\n"

        for distrib_field in distrib_fields:
            string += f"{distrib_field.upper()}:\n"
            for distrib in simple_data[distrib_field]:
                distrib_str_arr = [f"{k}:{v}" for k, v in distrib.items()]
                string += f"\t{', '.join(distrib_str_arr)}\n"

        return string

    @classmethod
    def _extract_very_basic_fields(cls, data: dict) -> dict:
        # extract only name, keyword, description and platform
        new_object = {
            "id": data["identifier"],
            "platform": data["platform"],
            "name": data["name"],
        }
        description = cls._get_text_like_field(data, "description")
        if description is not None:
            new_object["description"] = description

        keywords = data.get("keyword", [])
        if len(keywords) > 0:
            new_object["keyword"] = keywords

        return new_object

    @classmethod
    def _extract_relevant_fields(cls, data: dict, asset_type: SupportedAssetType) -> dict:
        new_object = {}

        flat_fields = cls.orig_flat_fields
        if asset_type == SupportedAssetType.EDUCATIONAL_RESOURCES:
            flat_fields.extend(cls.educational_flat_fields)
        elif asset_type == SupportedAssetType.EXPERIMENTS:
            flat_fields.extend(cls.experiment_flat_fields)
        elif asset_type == SupportedAssetType.SERVICES:
            flat_fields.extend(cls.service_flat_fields)

        for field_name in flat_fields:
            x = data.get(field_name, None)
            if x is not None:
                new_object[field_name] = x

        array_fields = cls.orig_array_fields
        if asset_type == SupportedAssetType.EDUCATIONAL_RESOURCES:
            array_fields.extend(cls.educational_array_fields)
        for field_name in array_fields:
            x = data.get(field_name, None)
            if x is not None and len(x) > 0:
                new_object[field_name] = x

        # description & content
        for field in ["description", "content"]:
            val = cls._get_text_like_field(data, field)
            if val is not None:
                new_object[field] = val

        # Distribution type data (fields: distribution, media)
        dist_relevant_fields = [
            "name",
            "description",
            "content_size_kb",
            "encoding_format",
        ]
        if asset_type == SupportedAssetType.ML_MODELS:
            dist_relevant_fields.extend(["hardware_requirement", "os_requirement"])
        elif asset_type == SupportedAssetType.EXPERIMENTS:
            dist_relevant_fields.extend(
                [
                    "hardware_requirement",
                    "os_requirement",
                    "installation",
                    "deployment",
                    "dependency",
                ]
            )
        for field_name in cls.orig_distrib_fields:
            distrib_list = data.get(field_name, None)
            if distrib_list is not None and len(distrib_list) > 0:
                new_object[field_name] = []

                for distrib in distrib_list:
                    new_dist = {k: distrib[k] for k in dist_relevant_fields if k in distrib}

                    if new_dist.get("content_size_kb", None) is not None:
                        size_kb = int(new_dist["content_size_kb"])
                        new_dist["content_size_mb"] = round(size_kb / 1024, 2)
                        new_dist["content_size_gb"] = round(size_kb / 1024**2, 2)
                    if new_dist != {}:
                        new_object[field_name].append(new_dist)

        # Note
        notes = data.get("note", None)
        if notes is not None and len(notes) > 0:
            new_object["note"] = [note["value"] for note in notes if "value" in note]

        # Size
        if asset_type == SupportedAssetType.DATASETS:
            size = data.get("size", None)
            if size is not None and "unit" in size and "value" in size:
                new_object["size"] = f"{size['value']} {size['unit']}"

        return new_object

    @classmethod
    def _get_text_like_field(cls, data: dict, field: str) -> str | None:
        # TODO: Simplify this logic
        description = data.get(field, None)
        if description is None:
            return None

        plain_descr = description.get("plain", None)
        html_descr = description.get("html", None)
        if html_descr is not None:
            html_descr = re.sub(r"<[^>]*>", " ", html_descr)

        if plain_descr is not None and html_descr is not None:
            return f"{plain_descr} {html_descr}"
        if plain_descr is not None:
            return plain_descr
        if html_descr is not None:
            return html_descr
        return None
