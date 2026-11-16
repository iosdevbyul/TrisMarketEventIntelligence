"""SEC company-history archive inventory uses source hashes and quarantines clocks."""
from __future__ import annotations

import json
from datetime import date

import httpx
import pytest

from tris_market_event_intelligence.sec_historical_filing_inventory_cli import (
    _get_json,
    build_sec_filing_inventory,
    inventory_from_sec,
)

CIK = "0001045810"


def _columns() -> dict:
    return {
        "accessionNumber": [
            "0001045810-18-000148",
            "0001045810-18-000149",
            "0001045810-23-000093",
        ],
        "filingDate": ["2018-11-15", "2018-11-19", "2023-05-24"],
        "acceptanceDateTime": [
            "2018-11-15T16:21:48.000Z",
            "2018-11-19T17:22:00.000Z",
            "2023-05-24T16:33:00.000Z",
        ],
        "form": ["8-K", "10-Q", "8-K"],
        "primaryDocument": ["earnings.htm", "quarterly.htm", "nvda.htm"],
        "items": ["2.02,9.01", "", "2.02"],
    }


def _main() -> dict:
    return {
        "cik": 1045810,
        "filings": {
            "recent": _columns(),
            "files": [{
                "name": f"CIK{CIK}-submissions-001.json",
                "filingFrom": "2016-01-01",
                "filingTo": "2017-12-31",
            }],
        },
    }


def test_candidates_are_not_approved_historic_news() -> None:
    report = build_sec_filing_inventory(
        main=_main(), main_sha="a" * 64, histories=[],
        start=date(2018, 11, 15), end=date(2023, 5, 25),
    )
    assert report["candidateFilings"] == 3
    assert report["formCounts"] == {"10-Q": 1, "8-K": 2}
    assert report["trainingReady"] is False
    row = report["filings"][0]
    assert row["rawSecAcceptanceDateTime"] == "2018-11-15T16:21:48.000Z"
    assert row["eligibleAsNewsModelFeature"] is False
    assert row["secAcceptanceTimeIndependentlyVerified"] is False
    assert row["secIndexUrl"].endswith(
        "/104581018000148/0001045810-18-000148-index.htm"
    )
    assert "direction" not in json.dumps(row).lower()


def test_official_history_fetch_only_overlap_and_source_hash() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path.endswith(f"CIK{CIK}.json"):
            return httpx.Response(200, json=_main())
        if request.url.path.endswith("-001.json"):
            return httpx.Response(200, json={
                **_columns(),
                "accessionNumber": [
                    "0001045810-17-000001",
                    "0001045810-17-000002",
                    "0001045810-17-000003",
                ],
                "filingDate": ["2017-05-09", "2017-05-10", "2017-11-20"],
            })
        raise AssertionError("Unexpected network URL")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = inventory_from_sec(
            client=client,
            user_agent="TrisMarketEventIntelligence contact@example.com",
            start=date(2017, 1, 1),
            end=date(2018, 12, 31),
        )
    assert len(calls) == 2
    assert all("data.sec.gov/submissions" in item for item in calls)
    assert result["sourcePages"] == 2
    assert result["candidateFilings"] == 5
    assert result["trainingReady"] is False
    assert len(result["sources"][1]["sha256"]) == 64


def test_no_undisclosed_sec_bot_identification() -> None:
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: pytest.fail("Network attempted without User-Agent")
    )) as client:
        with pytest.raises(ValueError, match="User-Agent"):
            inventory_from_sec(
                client=client,
                user_agent="",
                start=date(2018, 1, 1),
                end=date(2018, 12, 31),
            )


def test_bad_parallel_columns_and_nonhistorical_ranges_fail() -> None:
    bad = _main()
    bad["filings"]["recent"]["acceptanceDateTime"].pop()
    with pytest.raises(ValueError, match="parallel columns"):
        build_sec_filing_inventory(
            main=bad, main_sha="a" * 64, histories=[],
            start=date(2018, 1, 1), end=date(2018, 12, 31),
        )
    with pytest.raises(ValueError, match="scope"):
        build_sec_filing_inventory(
            main=_main(), main_sha="a" * 64, histories=[],
            start=date(2000, 1, 1), end=date(2026, 11, 16),
        )


def test_prevent_traversal_and_bad_download_host() -> None:
    bad = _main()
    bad["filings"]["recent"]["primaryDocument"][0] = "../../secrets"
    with pytest.raises(ValueError, match="traversal"):
        build_sec_filing_inventory(
            main=bad, main_sha="a" * 64, histories=[],
            start=date(2018, 1, 1), end=date(2018, 12, 31),
        )
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: pytest.fail("Unexpected domain")
    )) as client:
        with pytest.raises(ValueError, match="Unapproved SEC"):
            _get_json(client, "https://example.com/untrusted", "abc@example.com")
