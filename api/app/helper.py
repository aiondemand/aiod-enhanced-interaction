from datetime import datetime, timezone
from typing import Literal


def parse_asset_date(
    asset: dict,
    field: str = "date_modified",
    none_value: Literal["none", "now", "zero"] = "none",
) -> datetime | None:
    string_time = asset.get("aiod_entry", {}).get(field, None)
    if string_time is None:
        if none_value == "none":
            return None
        if none_value == "now":
            return datetime.now(tz=timezone.utc)
        if none_value == "zero":
            return datetime.fromtimestamp(0, tz=timezone)

    return datetime.fromisoformat(string_time).replace(tzinfo=timezone.utc)
