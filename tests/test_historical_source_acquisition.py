"""Provider-independent source acquisition: real payload contracts, no invented data."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import date
from pathlib import Path

import httpx
import pytest

from tris_market_event_intelligence.historical_source_acquisition import (
    HistoricalProviderError,
    create_private_artifact,
    fred_revision_archive,
    json_bytes,
    tiingo_eod,
    verify_tiingo_export,
)


def test_tiingo_real_payload_uses_adjusted_close_and_private_export(tmp_path: Path) -> None:
    requests = []
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["authorization"] == "Token SECRET"
        assert request.url.path == "/tiingo/daily/NVDA/prices"
        assert request.url.params["startDate"] == "2026-10-01"
        return httpx.Response(200, json=[
            {"date": f"2026-10-0{i}T00:00:00.000Z",
             "adjClose": 100 + i, "close": 300 + i,
             "volume": 12345 + i}
            for i in (1, 2, 5, 6, 7)
        ])
    with httpx.Client(transport=httpx.MockTransport(handler)) as session:
        raw, metadata = tiingo_eod(
            symbol="NVDA", start=date(2026, 10, 1), end=date(2026, 10, 8),
            token="SECRET", client=session,
        )
    csv_path = tmp_path / "nvda.csv"
    digest = create_private_artifact(csv_path, raw)
    assert digest == hashlib.sha256(raw).hexdigest()
    assert metadata["sha256"] == digest
    rows = list(csv.DictReader(raw.decode().splitlines()))
    assert rows[0]["adjusted_close"] == "101"
    assert rows[0]["volume"] == "12346"
    assert verify_tiingo_export(csv_path, digest) == 5
    assert metadata["providerAuthenticityIndependentlyVerified"] is False
    assert metadata["trainingReady"] is False
    assert len(requests) == 1
    assert "SECRET" not in raw.decode()
    with pytest.raises(HistoricalProviderError, match="PRIVATE_DESTINATION_INVALID"):
        create_private_artifact(csv_path, raw)


def test_fred_archive_keeps_revision_dates_and_never_fakes_release_clock(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.stlouisfed.org"
        assert request.url.params["realtime_start"] == "2026-01-01"
        assert request.url.params["realtime_end"] == "2026-12-31"
        return httpx.Response(200, json={
            "count": 3,
            "observations": [
                {"date": "2026-01-01", "realtime_start": "2026-02-12",
                 "realtime_end": "2026-03-09", "value": "3.2"},
                {"date": "2026-01-01", "realtime_start": "2026-03-10",
                 "realtime_end": "2026-12-31", "value": "3.1"},
                {"date": "2026-02-01", "realtime_start": "2026-03-10",
                 "realtime_end": "2026-12-31", "value": "."},
            ],
        })
    with httpx.Client(transport=httpx.MockTransport(handler)) as session:
        result = fred_revision_archive(
            series_id="CPIAUCSL", start=date(2026, 1, 1), end=date(2026, 12, 31),
            api_key="a" * 32, client=session,
        )
    assert result["rows"] == 2
    assert result["observations"][0]["value"] == 3.2
    assert result["observations"][1]["value"] == 3.1
    assert result["observations"][1]["firstKnownDate"] == "2026-03-10"
    assert result["releaseClockVerified"] is False
    assert result["pointInTimeIntradayEligible"] is False
    assert result["trainingReady"] is False
    assert "releaseAt" not in json.dumps(result)
    target = tmp_path / "fred.json"
    create_private_artifact(target, json_bytes(result))
    assert json.loads(target.read_text())["observations"] == result["observations"]


@pytest.mark.parametrize("record", [
    {"date": "2026-10-01T00:00:00Z", "adjClose": -3, "volume": 3},
    {"date": "2026-10-01T00:00:00Z", "adjClose": 10, "volume": -1},
    {"date": "2026-10-01T00:00:00Z", "adjClose": 10},
])
def test_price_rejects_invalid_provider_rows(record: dict) -> None:
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=[record] * 4),
    )) as session:
        with pytest.raises(HistoricalProviderError):
            tiingo_eod(
                symbol="NVDA", start=date(2026, 10, 1),
                end=date(2026, 10, 9), token="SECRET", client=session,
            )


def test_provider_error_does_not_leak_http_credential() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"message": "private token SECRET"})
    with httpx.Client(transport=httpx.MockTransport(handler)) as session:
        with pytest.raises(HistoricalProviderError) as failure:
            tiingo_eod(
                symbol="NVDA", start=date(2026, 10, 1),
                end=date(2026, 10, 9), token="SECRET", client=session,
            )
    assert failure.value.args == ("PROVIDER_FETCH_OR_JSON_FAILED",)


def test_no_macro_key_fails_before_network() -> None:
    calls = []
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={})
    with httpx.Client(transport=httpx.MockTransport(handler)) as session:
        with pytest.raises(HistoricalProviderError, match="FRED_API_KEY_REQUIRED"):
            fred_revision_archive(
                series_id="CPIAUCSL", start=date(2026, 1, 1),
                end=date(2026, 12, 31), api_key="", client=session,
            )
    assert not calls
