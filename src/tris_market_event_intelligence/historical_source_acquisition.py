"""Explicit opt-in acquisition of real licensed price and macro-vintage evidence.

No downloads on import, and no automatic training or 24/7 activity.
FRED revision dates are NOT release clock-times: never synthesize those.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
from datetime import date
from math import isfinite
from pathlib import Path

import httpx

from tris_market_event_intelligence.historical_shock_research import read_daily_prices

_SYMBOL = re.compile(r"[A-Z0-9][A-Z0-9._-]{0,19}\Z")
_SERIES = re.compile(r"[A-Za-z0-9_]{1,48}\Z")
_MAX_RESPONSE = 8_000_000
_MAX_OBSERVATIONS = 50000


class HistoricalProviderError(ValueError):
    """Stable error codes; never disclose credentials or raw response bodies."""


def _validate_dates(start: date, end: date) -> None:
    if not isinstance(start, date) or not isinstance(end, date) or not start < end:
        raise HistoricalProviderError("INVALID_DATE_WINDOW")
    if (end - start).days > 365 * 30:
        raise HistoricalProviderError("WINDOW_TOO_LARGE")


def _get_json(
    session: httpx.Client, endpoint: str, *,
    params: dict[str, str], headers: dict[str, str] | None = None,
) -> object:
    try:
        response = session.get(endpoint, params=params, headers=headers or {})
        response.raise_for_status()
        if len(response.content) > _MAX_RESPONSE:
            raise HistoricalProviderError("PROVIDER_RESPONSE_OVERSIZED")
        return response.json()
    except HistoricalProviderError:
        raise
    except (httpx.HTTPError, ValueError) as exc:
        raise HistoricalProviderError("PROVIDER_FETCH_OR_JSON_FAILED") from exc


def tiingo_eod(
    *, symbol: str, start: date, end: date,
    token: str, client: httpx.Client,
) -> tuple[bytes, dict[str, object]]:
    """Fetch 10y adjusted EOD from provider; require actual adequate coverage."""
    _validate_dates(start, end)
    if not _SYMBOL.fullmatch(symbol):
        raise HistoricalProviderError("INVALID_SYMBOL")
    if not token or len(token) > 256:
        raise HistoricalProviderError("TIINGO_TOKEN_REQUIRED")
    payload = _get_json(
        client, f"https://api.tiingo.com/tiingo/daily/{symbol}/prices",
        params={"startDate": start.isoformat(), "endDate": end.isoformat()},
        headers={"Authorization": "Token " + token},
    )
    if not isinstance(payload, list) or not 4 <= len(payload) <= 8000:
        raise HistoricalProviderError("PRICE_HISTORY_INSUFFICIENT_OR_INVALID")
    records: list[tuple[date, float, int]] = []
    for row in payload:
        if not isinstance(row, dict):
            raise HistoricalProviderError("PRICE_ROW_INVALID")
        try:
            raw_date = row["date"]
            if not isinstance(raw_date, str):
                raise ValueError("invalid date")
            day = date.fromisoformat(raw_date[:10])
            price = float(row["adjClose"])
            volume = row["volume"]
            if (isinstance(volume, bool) or type(volume) not in (int, float)
                    or not isfinite(float(volume)) or volume < 0
                    or not float(volume).is_integer()
                    or not isfinite(price) or price <= 0
                    or not start <= day <= end):
                raise ValueError("invalid numeric price")
            records.append((day, price, int(volume)))
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            raise HistoricalProviderError("PRICE_ROW_INVALID") from exc
    records.sort(key=lambda item: item[0])
    if len({day for day, _, _ in records}) != len(records):
        raise HistoricalProviderError("DUPLICATE_DAILY_PRICE")
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["date", "adjusted_close", "volume"])
    for day, price, volume in records:
        writer.writerow([day.isoformat(), f"{price:.12g}", volume])
    raw = output.getvalue().encode("utf-8")
    return raw, {
        "schemaVersion": "authorized_price_acquisition_v1",
        "provider": "tiingo-eod", "symbol": symbol,
        "firstDate": records[0][0].isoformat(),
        "lastDate": records[-1][0].isoformat(),
        "dailySessions": len(records),
        "requestedStart": start.isoformat(), "requestedEnd": end.isoformat(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "adjustedCloseFromProvider": True,
        "providerAuthenticityIndependentlyVerified": False,
        "dataLicenceApprovedForUse": False,
        "trainingReady": False,
        "newsOrMacroUsed": False,
    }


def fred_revision_archive(
    *, series_id: str, start: date, end: date,
    api_key: str, client: httpx.Client,
) -> dict[str, object]:
    """Capture ALFRED-style real-time observations, no fictitious release_at."""
    _validate_dates(start, end)
    if not _SERIES.fullmatch(series_id):
        raise HistoricalProviderError("INVALID_MACRO_SERIES")
    if not re.fullmatch(r"[a-z0-9]{32}", api_key):
        raise HistoricalProviderError("FRED_API_KEY_REQUIRED")
    # Uses the documented full real-time interval and its date-granular
    # observation revisions, not today's revised/latest time series.
    payload = _get_json(
        client, "https://api.stlouisfed.org/fred/series/observations",
        params={
            "series_id": series_id, "api_key": api_key,
            "file_type": "json",
            "realtime_start": start.isoformat(),
            "realtime_end": end.isoformat(),
            "observation_start": start.isoformat(),
            "observation_end": end.isoformat(),
            "limit": "100000", "output_type": "1",
        },
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("observations"), list):
        raise HistoricalProviderError("MACRO_ARCHIVE_INVALID")
    if payload.get("count", 0) > _MAX_OBSERVATIONS:
        raise HistoricalProviderError("MACRO_ARCHIVE_TOO_LARGE")
    items: list[dict[str, object]] = []
    for row in payload["observations"]:
        if not isinstance(row, dict):
            raise HistoricalProviderError("MACRO_ROW_INVALID")
        try:
            observation_date = date.fromisoformat(row["date"])
            first_known_date = date.fromisoformat(row["realtime_start"])
            last_known_date = date.fromisoformat(row["realtime_end"])
            value_text = row["value"]
            if not isinstance(value_text, str):
                raise ValueError("invalid value")
            if value_text == ".":
                continue
            value = float(value_text)
            if (not isfinite(value) or last_known_date < first_known_date
                    or not start <= first_known_date <= end
                    or not start <= observation_date <= end):
                raise ValueError("invalid value or revision dates")
            items.append({
                "observationDate": observation_date.isoformat(),
                "firstKnownDate": first_known_date.isoformat(),
                "lastKnownDate": last_known_date.isoformat(),
                "value": value,
            })
        except (KeyError, ValueError, TypeError, OverflowError) as exc:
            raise HistoricalProviderError("MACRO_ROW_INVALID") from exc
    if len(items) > _MAX_OBSERVATIONS:
        raise HistoricalProviderError("MACRO_ARCHIVE_TOO_LARGE")
    return {
        "schemaVersion": "fred_macro_date_vintage_evidence_v1",
        "source": "FRED/ALFRED",
        "seriesId": series_id, "requestedStart": start.isoformat(),
        "requestedEnd": end.isoformat(), "rows": len(items),
        "observations": items,
        "releaseClockVerified": False,
        "availableAtVerified": False,
        "pointInTimeIntradayEligible": False,
        "requiresIndependentReleaseClockEvidence": True,
        "sourceAuthenticityIndependentlyVerified": False,
        "trainingReady": False,
    }


def create_private_artifact(path: Path, raw: bytes) -> str:
    """Create-only, symlink-refusing and 0600 private research artifact."""
    if (path.exists() or path.is_symlink() or not path.parent.is_dir()
            or path.parent.is_symlink()):
        raise HistoricalProviderError("PRIVATE_DESTINATION_INVALID")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as out:
            out.write(raw)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return hashlib.sha256(raw).hexdigest()


def verify_tiingo_export(path: Path, expected_sha: str) -> int:
    """Validate the generated export using existing Phase 1/2 contract."""
    bars, sha = read_daily_prices(path)
    if sha != expected_sha:
        raise HistoricalProviderError("EXPORTED_PRICE_SHA_MISMATCH")
    return len(bars)


def json_bytes(obj: dict[str, object]) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
