from datetime import datetime


def translate_datetime_to_aiod_params(date: datetime) -> str | None:
    if date is None:
        return date
    return f"{date.year}-{date.month}-{date.day}"
