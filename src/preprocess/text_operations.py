from datetime import datetime
import json
import re


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

    @classmethod
    def extract_very_basic_info(cls, data: dict, stringify: bool = False) -> str:
        simple_data = cls._extract_very_basic_fields(data)
        if stringify:
            return json.dumps(simple_data)
        
        description = simple_data.get("description", None)
        keywords = simple_data.get("keyword", None)

        string = f"Platform: {simple_data['platform']}\nAsset name: {simple_data['name']}"
        if description is not None:
            string += f"\nDescription: {description}"
        if keywords is not None:
            key_string = " | ".join(keywords)
            string += f"\nKeywords: {key_string}"

        return string
    
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