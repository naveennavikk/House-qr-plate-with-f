"""Fetches house records (with both photos) from the ICTSBM house details API.

Each record looks like:
  {"userId": 5, "houseId": 7044, "ReferanceId": "HPSBA-G8044",
   "QRCodeImage": "data:image/jpeg;base64,...",   # reference QR plate photo
   "HouseImage":  "data:image/jpeg;base64,...",   # property photo
   "QRStatus": null, "Remark": null, "qrEmpName": "...", "modified": "..."}

The API is a GET whose parameters are all request headers:
  AppId: 3098, UserId: -1, FromDate: 2025-06-01, ToDate: 2026-11-23, SearchString: ""

Settings (environment variables):
  HOUSE_SOURCE_URL          API address (default: GetHouseDetailsByDate below)
  HOUSE_SOURCE_APP_ID       AppId header used when no ULB is chosen (default 3098 = IT Department)
  ULB_LIST_URL              list of ULBs [{"ulb": name, "appid": id}] for the page's ULB dropdown
  HOUSE_SOURCE_USER_ID      UserId header (default -1 = all users)
  HOUSE_SOURCE_HEADERS      JSON object of any further headers, e.g. '{"SearchString": "..."}'
  HOUSE_SOURCE_DATE_FORMAT  strftime format of FromDate/ToDate (default "%Y-%m-%d")
  HOUSE_SOURCE_TIMEOUT      seconds to wait for the API (default 600)
"""

import http.client
import json
import os
import time
import urllib.error
import urllib.request
from datetime import date

SOURCE_URL = os.getenv("HOUSE_SOURCE_URL", "https://weight.ictsbm.com/api/Get/GetHouseDetailsByDate")
APP_ID = os.getenv("HOUSE_SOURCE_APP_ID", "3098")
ULB_LIST_URL = os.getenv("ULB_LIST_URL", "https://weight.ictsbm.com/api/Get/AllULBListForWeightCollection")
ULB_CACHE_SECONDS = 600
_ulb_cache: tuple[float, list[dict]] | None = None
USER_ID = os.getenv("HOUSE_SOURCE_USER_ID", "-1")
DATE_FORMAT = os.getenv("HOUSE_SOURCE_DATE_FORMAT", "%Y-%m-%d")
TIMEOUT = float(os.getenv("HOUSE_SOURCE_TIMEOUT", "600"))


class SourceError(Exception):
    """The house API could not be reached or returned something unusable."""


def _headers() -> dict:
    raw = os.getenv("HOUSE_SOURCE_HEADERS", "").strip()
    if not raw:
        return {}
    try:
        headers = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SourceError(f"HOUSE_SOURCE_HEADERS is not valid JSON: {exc}") from None
    return {str(k): str(v) for k, v in headers.items()}


def _records(payload) -> list[dict]:
    """The list of house records, whether the API returns a list or wraps it in an object."""
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        if "houseId" in payload or "HouseImage" in payload:
            return [payload]
        for value in payload.values():
            if isinstance(value, list) and all(isinstance(r, dict) for r in value):
                return value
        if "error" in payload or "message" in payload:
            raise SourceError(str(payload.get("error") or payload.get("message")))
    raise SourceError("The house API returned data in an unexpected format.")


def fetch_ulbs() -> list[dict]:
    """ULBs (urban local bodies) as [{"name", "appId"}], sorted by name; cached for 10 minutes."""
    global _ulb_cache
    if _ulb_cache and time.monotonic() - _ulb_cache[0] < ULB_CACHE_SECONDS:
        return _ulb_cache[1]
    request = urllib.request.Request(ULB_LIST_URL, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed, configured URL
            payload = json.loads(response.read())
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        raise SourceError(f"Could not load the ULB list: {exc}") from None
    ulbs = sorted(
        ({"name": str(r["ulb"]).strip(), "appId": int(r["appid"])} for r in _records(payload) if r.get("ulb") and r.get("appid")),
        key=lambda u: u["name"].lower(),
    )
    _ulb_cache = (time.monotonic(), ulbs)
    return ulbs


def fetch_houses(from_day: date, to_day: date, search: str = "", app_id: int | str | None = None) -> list[dict]:
    headers = {
        "Accept": "application/json",
        "AppId": str(app_id or APP_ID),
        "UserId": USER_ID,
        "FromDate": from_day.strftime(DATE_FORMAT),
        "ToDate": to_day.strftime(DATE_FORMAT),
        "SearchString": search,
        **_headers(),
    }
    request = urllib.request.Request(SOURCE_URL, headers=headers)
    range_text = f"{headers['FromDate']} to {headers['ToDate']}"
    for attempt in (1, 2):
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:  # noqa: S310 - fixed, configured URL
                body = response.read()
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:500].decode("utf-8", "replace")
            try:
                detail = json.loads(detail).get("error", detail)
            except (ValueError, AttributeError):
                pass
            raise SourceError(f"House API returned HTTP {exc.code}: {detail}") from None
        except (http.client.RemoteDisconnected, ConnectionResetError, http.client.IncompleteRead) as exc:
            # The house API's server drops the connection when its query runs too long.
            waited = round(time.monotonic() - started)
            if attempt == 2:
                raise SourceError(
                    f"The house API closed the connection without sending data after {waited} s "
                    f"(tried twice, {range_text}). Its database is probably too slow for this "
                    f"date range: try a shorter range or try again later. ({type(exc).__name__})"
                ) from None
        except TimeoutError:
            raise SourceError(f"The house API did not answer within {int(TIMEOUT)} s ({range_text}).") from None
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            raise SourceError(f"Could not reach the house API: {exc}") from None
    try:
        return _records(json.loads(body))
    except json.JSONDecodeError:
        raise SourceError("The house API did not return JSON.") from None
