import logging
from datetime import datetime
from time import sleep

import requests
from requests import Response
from requests.exceptions import HTTPError, Timeout

from app.config import settings
from app.schemas.enums import AssetType
from app.schemas.request_params import RequestParams


def recursive_aiod_asset_fetch(
    asset_type: AssetType, url_params: RequestParams, mark_recursions: list[int]
) -> list:
    sleep(settings.AIOD.JOB_WAIT_INBETWEEN_REQUESTS_SEC)
    url = settings.AIOD.get_assets_url(asset_type)
    queries = _build_aiod_url_queries(url_params)

    try:
        response = perform_url_request(url, queries)
        data = response.json()
    except HTTPError as e:
        if e.response.status_code != 500:
            raise e
        elif url_params.limit == 1:
            return []

        mark_recursions.append(1)
        first_half_limit = url_params.limit // 2
        second_half_limit = url_params.limit - first_half_limit

        first_half = recursive_aiod_asset_fetch(
            url, url_params.new_page(limit=first_half_limit), mark_recursions
        )
        second_half = recursive_aiod_asset_fetch(
            url,
            url_params.new_page(
                offset=url_params.offset + first_half_limit, limit=second_half_limit
            ),
            mark_recursions,
        )
        return first_half + second_half

    return data


def get_aiod_document(doc_id: str, asset_type: AssetType, sleep_time: float = 0.1) -> dict | None:
    try:
        sleep(sleep_time)
        response = perform_url_request(settings.AIOD.get_asset_by_id_url(doc_id, asset_type))
        return response.json()
    except HTTPError as e:
        if e.response.status_code == 404:
            return None
        raise


def check_aiod_document(doc_id: str, asset_type: AssetType, sleep_time: float = 0.1) -> bool:
    return get_aiod_document(doc_id, asset_type, sleep_time) is not None


def _build_aiod_url_queries(url_params: RequestParams) -> dict:
    def translate_datetime_to_aiod_params(date: datetime | None = None) -> str | None:
        if date is None:
            return date
        return f"{date.year}-{date.month}-{date.day}"

    return {
        "schema": "aiod",
        "offset": url_params.offset,
        "limit": url_params.limit,
        "date_modified_after": translate_datetime_to_aiod_params(url_params.from_time),
        "date_modified_before": translate_datetime_to_aiod_params(url_params.to_time),
    }


def perform_url_request(
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
