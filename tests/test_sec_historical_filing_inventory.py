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
        "/000104581018000148/0001045810-18-000148-index.htm"
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
    )) as client, pytest.raises(ValueError, match="User-Agent"):
        inventory_from_sec(
            client=client,
            user_agent="",
            start=date(2018, 1, 1),
            end=date(2018, 12, 31),
        )


def test_bad_parallel_columns_and_nonhistorical_ranges_fail() -> None:
    bad = _main()
    bad["filings"]["recent"]["acceptanceDateTime"].pop()
    with pytest.raises(ValueError, match="SEC_SCHEMA_COLUMNS"):
        build_sec_filing_inventory(
            main=bad, main_sha="a" * 64, histories=[],
            start=date(2018, 1, 1), end=date(2018, 12, 31),
        )
    with pytest.raises(ValueError, match="SEC_SCOPE_OR_CIK"):
        build_sec_filing_inventory(
            main=_main(), main_sha="a" * 64, histories=[],
            start=date(2000, 1, 1), end=date(2026, 11, 16),
        )


def test_prevent_traversal_and_bad_download_host() -> None:
    bad = _main()
    bad["filings"]["recent"]["primaryDocument"][0] = "../../secrets"
    with pytest.raises(ValueError, match="SEC_PRIMARY_DOCUMENT_PATH"):
        build_sec_filing_inventory(
            main=bad, main_sha="a" * 64, histories=[],
            start=date(2018, 1, 1), end=date(2018, 12, 31),
        )
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: pytest.fail("Unexpected domain")
    )) as client, pytest.raises(ValueError, match="SEC_UNAPPROVED_SOURCE_URL"):
        _get_json(client, "https://example.com/untrusted", "abc@example.com")


def test_nullable_or_missing_items_are_quarantined_not_rejected() -> None:
    columns = _columns()
    columns["items"][0] = None
    main = _main()
    main["filings"]["recent"] = columns
    report = build_sec_filing_inventory(
        main=main, main_sha="a" * 64, histories=[],
        start=date(2018, 1, 1), end=date(2023, 12, 31),
    )
    assert report["filings"][0]["formItems"] == ""
    assert report["filings"][0]["eligibleAsNewsModelFeature"] is False
    del columns["items"]
    report = build_sec_filing_inventory(
        main=main, main_sha="a" * 64, histories=[],
        start=date(2018, 1, 1), end=date(2023, 12, 31),
    )
    assert report["candidateFilings"] == 3
    assert all(row["formItems"] == "" for row in report["filings"])


def test_sec_api_error_keeps_status_code_available_without_body() -> None:
    def deny(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403, text="Access denied; private user agent or email was sent"
        )

    with (
        httpx.Client(transport=httpx.MockTransport(deny)) as client,
        pytest.raises(httpx.HTTPStatusError) as exc,
    ):
        _get_json(
            client, f"https://data.sec.gov/submissions/CIK{CIK}.json",
            "TrisMarketEventIntelligence contact@example.com",
        )
    assert exc.value.response.status_code == 403


def test_history_overlap_does_not_confuse_provenance_with_filing_data() -> None:
    main = _main()
    history = _columns()
    # A recent/history pagination overlap is valid even with distinct
    # original page SHA and page names.
    doc = build_sec_filing_inventory(
        main=main,
        main_sha="a" * 64,
        histories=[(f"CIK{CIK}-submissions-001.json", history, "b" * 64)],
        start=date(2018, 1, 1),
        end=date(2023, 12, 31),
    )
    assert doc["candidateFilings"] == 3
    assert doc["sourcePages"] == 2
    assert doc["filings"][0]["submissionSourcePageSha256"] == "a" * 64


def test_real_filing_disagreement_still_fails_closed() -> None:
    history = _columns()
    history["form"][0] = "10-K"
    with pytest.raises(ValueError, match="SEC_ACCESSION_CONFLICT"):
        build_sec_filing_inventory(
            main=_main(),
            main_sha="a" * 64,
            histories=[(f"CIK{CIK}-submissions-001.json", history, "b" * 64)],
            start=date(2018, 1, 1),
            end=date(2023, 12, 31),
        )


def test_safe_nested_sec_primary_document_path() -> None:
    main = _main()
    main["filings"]["recent"]["primaryDocument"][0] = "reports/earnings.htm"
    result = build_sec_filing_inventory(
        main=main, main_sha="a" * 64, histories=[],
        start=date(2018, 1, 1), end=date(2023, 12, 31),
    )
    assert result["filings"][0]["secPrimaryDocument"] == "reports/earnings.htm"
    main["filings"]["recent"]["primaryDocument"][0] = "reports/../secret"
    with pytest.raises(ValueError, match="SEC_PRIMARY_DOCUMENT_PATH"):
        build_sec_filing_inventory(
            main=main, main_sha="a" * 64, histories=[],
            start=date(2018, 1, 1), end=date(2023, 12, 31),
        )
