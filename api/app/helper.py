import logging
from datetime import datetime, timezone
from time import sleep
from typing import Literal

import requests
from requests import Response
from requests.exceptions import Timeout


def translate_datetime_to_aiod_params(date: datetime) -> str | None:
    if date is None:
        return date
    return f"{date.year}-{date.month}-{date.day}"


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


def _perform_request(
    url: str,
    params: dict | None = None,
    num_retries: int = 3,
    connection_timeout_sec: int = 30,
) -> Response:
    # timeout based on the size of the requested response
    # 1000 assets == additional 1 minute of timeout
    limit = 0 if params is None else params.get("limit", 0)
    request_timeout = connection_timeout_sec + int(limit * 0.06)

    for _ in range(num_retries):
        try:
            response = requests.get(url, params, timeout=request_timeout)
            response.raise_for_status()
            return response
        except Timeout:
            logging.warning("AIoD endpoints are unresponsive. Retrying...")
            sleep(connection_timeout_sec)

    # This exception will be only raised if we encounter exception
    # Timeout (ReadTimeout / ConnectionTimeout) consecutively for multiple times
    err_msg = "We couldn't connect to AIoD API"
    logging.error(err_msg)
    raise ValueError(err_msg)
