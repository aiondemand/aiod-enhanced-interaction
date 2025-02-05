from datetime import datetime
import json
import re
from typing import Any


class HuggingFaceDatasetExtractMedatada:    
    @classmethod
    def extract_hf_keywords(cls, asset: dict, keyword_type: str) -> list[str]:
        keywords = [
            (kw.split(":")[0], kw.split(":")[1]) 
            for kw in asset.get("keyword", [])
            if len(kw.split(":")) == 2
        ]
        return list(set([
            kw[1]
            for kw in keywords
            if kw[0] == keyword_type
        ]))
        
    @classmethod
    def strip_unknown(cls, values: list[str]) -> list[str]:
        return [val for val in values if val != "unknown"]

    @classmethod
    def filter_out_empty_fields(cls, obj: dict) -> dict:
        def not_empty(val: Any) -> bool:
            if val is None:
                return False
            if isinstance(val, list) or isinstance(val, str):
                return len(val) > 0
            return True
        
        return {
            k: v for k, v in obj.items()
            if not_empty(v)
        }
    
    @classmethod
    def simplify_license(cls, licenses: list[str]) -> str | None:    
        main_license_prefixes = sorted([
            "mit",
            "apache",
            "cc-by",
            "cc-by-sa",
            "cc-by-nc",
            "cc-by-nc-sa",
            "cc-by-nc-nd",
            "cc-by-nd",
            "bsd",
            "wtfpl",
            "llama",
            "cc0",
            "gpl",
            "agpl",
            "lgpl",
            "artistic",
            "afl",
            "cdla",
            "odc",
            "obdl",
            "pddl"
        ], key=len, reverse=True)
        openrail_license = "openrail"

        processed_licenses = []
        for lic in licenses:
            if openrail_license in lic:
                processed_licenses.append(openrail_license)
                continue
            for prefix in main_license_prefixes:
                if lic.startswith(prefix):
                    processed_licenses.append(prefix)
                    break
            
        if len(processed_licenses) == 0:
            return None
        return processed_licenses[0]

    @classmethod
    def translate_size_category(cls, size_categories: list[str]) -> tuple[int, int] | tuple[None, None]:
        substr = "<n<"
        units = { "k": 1_000, "m": 1_000_000, "b": 1_000_000_000, "t": 1_000_000_000_000 }
        
        size_categories = [
            cat for cat in size_categories
            if substr in cat or cat in ["n<1k", "n>1t"]
        ]
        for cat in size_categories:
            if cat == "n<1k":
                return 0, units["k"]
            if cat == "n>1t":
                return units["t"], units["t"] * 1_000

            lower_str = cat[:cat.find(substr)]
            upper_str = cat[cat.find(substr) + len(substr): ]

            bound_pattern = r"^(\d+[kmbt])|(\d+)$"
            if (
                bool(re.match(bound_pattern, lower_str)) and 
                bool(re.match(bound_pattern, upper_str))
            ):
                try:
                    lower_bound = int(lower_str)
                except:
                    lower_bound = int(lower_str[:-1]) * units[lower_str[-1]]

                try:
                    upper_bound = int(upper_str)
                except:
                    upper_bound = int(upper_str[:-1]) * units[upper_str[-1]]
            
                return lower_bound, upper_bound
        
        return None, None

    @classmethod
    def extract_huggingface_dataset_metadata(cls, obj: dict) -> dict:
        # For now we only support extracting of metadata information from HuggingFace
        # As other platforms have their metadata more spread out and we would need
        # an LLM to extract the same information that we can currently easily parse
        # from HuggingFace keywords
        if obj["platform"] != "huggingface":
            return {}

        date_published_str = (
            obj["date_published"] + "Z" 
            if obj.get("date_published", None) is not None 
            else None
        )

        distribs = obj.get("distribution", [])
        ds_size = int(sum([
            dist.get("content_size_kb", 0) / 1024
            for dist in distribs
        ]))
        ds_size = ds_size if ds_size is not None else None

        kw_licenses = cls.extract_hf_keywords(obj, keyword_type="license")
        if obj.get("license", None) is not None:
            kw_licenses = list(set([obj["license"]] + kw_licenses))
        license = cls.simplify_license(cls.strip_unknown(kw_licenses))

        task_types = cls.strip_unknown(list(set(
            cls.extract_hf_keywords(obj, keyword_type="task_categories") +
            cls.extract_hf_keywords(obj, keyword_type="task_ids")
        )))
        
        size_categories = cls.strip_unknown(cls.extract_hf_keywords(
            obj, keyword_type="size_categories"
        ))
        lower_bound, upper_bound = None, None
        if len(size_categories) > 0:
            lower_bound, upper_bound = cls.translate_size_category(size_categories)
        
        obj_to_return = {
            "date_published": date_published_str,
            "size_in_mb": ds_size,
            "license": license,
            "task_types": task_types,
            "languages": cls.extract_hf_keywords(obj, keyword_type="language"),
            "datapoints_lower_bound": lower_bound,
            "datapoints_upper_bound": upper_bound
        }
        return cls.filter_out_empty_fields(obj_to_return)


class ConvertJsonToString:
    orig_flat_fields = [
        "platform",
        "name",
        "date_published",
    ]
    orig_array_fields = [
        "keyword",
        "alternate_name",
        "application_area", 
        "industrial_sector", 
        "research_area", 
        "scientific_domain"
    ]
    orig_distrib_fields = [
        "distribution", 
        "media"
    ]
    
    new_flat_fields = [
        "year_published",
        "month_published",
        "day_published",
        "description",
        "size"
    ]
    new_array_fields = [
        "note"
    ]

    @classmethod
    def stringify(cls, data: dict) -> str:
        return json.dumps(data)

    # "BASIC INFO"
    @classmethod
    def extract_very_basic_info(
        cls, data: dict, 
        stringify: bool = False, 
        include_id: bool = False
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
    def extract_relevant_info(cls, data: dict, stringify: bool = False) -> str:
        simple_data = cls._extract_relevant_fields(data)
        if stringify:
            return json.dumps(simple_data)

        string = ""
        
        for flat_field in cls.orig_flat_fields + cls.new_flat_fields:
            if flat_field in simple_data:
                string += f"{flat_field}: {simple_data[flat_field]}\n"
        for array_field in cls.orig_array_fields + cls.new_array_fields:
            if array_field in simple_data:
                string += f"{array_field}: {', '.join(simple_data[array_field])}\n"
        for distrib_field in cls.orig_distrib_fields:
            if distrib_field in simple_data:
                string += f"{distrib_field.upper()}:\n"
                for distrib in simple_data[distrib_field]:
                    distrib_str_arr = [f"{k}:{v}" for k,v in distrib.items()]
                    string += f"\t{', '.join(distrib_str_arr)}\n"

        return string    
    
    @classmethod
    def _extract_very_basic_fields(cls, data: dict) -> dict:
        # extract only name, keyword, description and platform
        new_object = {
            "id": data["identifier"],
            "platform": data["platform"],
            "name": data["name"]
        }
        description = cls._get_description(data)
        if description is not None:
            new_object["description"] = description

        keywords = data.get("keyword", [])
        if len(keywords) > 0:
            new_object["keyword"] = keywords

        return new_object
            
    @classmethod
    def _extract_relevant_fields(cls, data: dict) -> dict:
        # basic fields to copy
        new_object = {}
        for field_value in cls.orig_flat_fields:
            x = data.get(field_value, None)
            if x is not None:
                new_object[field_value] = x

        if new_object.get("date_published", None) is not None:
            dt = datetime.fromisoformat(new_object["date_published"])
            new_object["year_published"] = dt.year
            new_object["month_published"] = dt.month
            new_object["day_published"] = dt.day

        for field_value in cls.orig_array_fields:
            x = data.get(field_value, None)
            if x is not None and len(x) > 0:
                new_object[field_value] = x
        
        # description
        description = cls._get_description(data)
        if description is not None:
            new_object["description"] = description
        
        # Distribution type data (fields: distribution, media)
        dist_relevant_fields = [
            "name", "description", "content_size_kb", "encoding_format"
        ] 
        for field_name in cls.orig_distrib_fields:
            field_value = data.get(field_name, None)
            if field_value is not None and len(field_value) > 0:
                new_object[field_name] = []
                
                for dist in field_value:
                    new_dist = {
                        k: dist[k]
                        for k in dist_relevant_fields
                        if k in dist                    
                    }
                    
                    if new_dist.get("content_size_kb", None) is not None:
                        size_kb = new_dist["content_size_kb"]
                        new_dist["content_size_mb"] = float(f"{(size_kb / 1024):.2f}")
                        new_dist["content_size_gb"] = float(f"{(size_kb / 1024**2):.2f}")
                    if new_dist != {}:
                        new_object[field_name].append(new_dist)
                
        # Note
        notes = data.get("note", None)
        if notes is not None and len(notes) > 0:
            new_object["note"] = [
                note["value"] 
                for note in notes 
                if "value" in note
            ]

        # Size
        size = data.get("size", None)
        if size is not None and "unit" in size and "value" in size:
            new_object["size"] = f"{size['value']} {size['unit']}"

        return new_object

    @classmethod
    def _get_description(cls, data: dict) -> str | None:
        description = data.get("description", None)
        if description is None:
            return None
        
        plain_descr = description.get("plain", None)
        html_descr = description.get("html", None)

        if plain_descr is not None:
            return plain_descr
        elif html_descr is not None:
            html_descr = re.sub(re.compile(r'<.*?>'), '', html_descr)
            return html_descr

        return None