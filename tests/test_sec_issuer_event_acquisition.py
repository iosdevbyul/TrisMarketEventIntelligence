"""Official SEC 8-K issuer-source backfill, validation and evidence-persistence."""
from __future__ import annotations

import sqlite3
from datetime import date

import httpx
import pytest

from tris_market_event_intelligence.market_event_candidate_policy import (
    research_candidate,
)
from tris_market_event_intelligence.sec_issuer_event_acquisition_cli import (
    collect_sec_filings,
    source_body,
    submission_candidates,
    verified_sec_acceptance,
)


def sec_recent(cik: int = 1045810) -> dict:
    return {
        "cik": cik, "name": "NVIDIA CORPORATION", "tickers": ["NVDA"],
        "filings": {"recent": {
            "accessionNumber": [
                "0001045810-26-000123", "0001045810-26-000124",
                "0001045810-26-000125",
            ],
            "form": ["8-K", "8-K", "10-Q"],
            "acceptanceDateTime": [
                "2026-08-19T18:32:10Z", "2026-09-01T20:00:00Z",
                "2026-09-02T21:00:00Z",
            ],
            "primaryDocument": ["nvda-8k.htm", "nvda-exhibit.htm", "nvda-10q.htm"],
            "items": ["2.02,9.01", "1.01,9.01", ""],
        }},
    }


def test_source_requires_offset_aware_actual_sec_acceptance_clock():
    assert verified_sec_acceptance("2026-08-19T18:32:10Z").isoformat() == (
        "2026-08-19T18:32:10+00:00"
    )
    with pytest.raises(ValueError, match="UNVERIFIED"):
        verified_sec_acceptance("2026-08-19T18:32:10")


def test_only_8k_earnings_and_material_contracts_with_valid_archive_url():
    candidates, reasons = submission_candidates(
        sec_recent(), cik=1045810, since=date(2026, 4, 1),
        until=date(2026, 11, 18),
    )
    assert len(candidates) == 2
    assert [x["eventCategory"] for x in candidates] == [
        "EARNINGS_8K", "MATERIAL_CONTRACT_8K",
    ]
    assert candidates[0]["sourceUrl"].endswith(
        "/1045810/000104581026000123/nvda-8k.htm"
    )
    assert candidates[0]["issuerReleaseTimeVerified"] is False
    assert reasons["NOT_FORM_8K"] == 1
    with pytest.raises(ValueError, match="SEC_CIK_RESPONSE_MISMATCH"):
        submission_candidates(
            sec_recent(), cik=2488, since=date(2026, 4, 1),
            until=date(2026, 11, 18),
        )


def test_sec_archive_path_traversal_rejected():
    payload = sec_recent()
    payload["filings"]["recent"]["primaryDocument"][0] = "../secrets.html"
    candidates, reasons = submission_candidates(
        payload, cik=1045810, since=date(2026, 4, 1),
        until=date(2026, 11, 18),
    )
    assert len(candidates) == 1
    assert reasons["INVALID_SEC_ARCHIVE_PATH"] == 1


def test_raw_filing_capture_writes_substantive_original_bytes_and_deduplicates(
    tmp_path,
):
    from datetime import datetime

    from tris_market_event_intelligence.collector import NewsStore

    db = tmp_path / "news.sqlite3"
    NewsStore(db).initialize()
    text = (
        "<html><body><h1>SEC FORM 8-K</h1><p>Item 2.02 Results of "
        "Operations and Financial Condition.</p><p>"
        + "The issuer reports its quarterly revenue, expenses and "
        "related financial results for the current reporting period. " * 10
        + "</p></body></html>"
    ).encode()
    called = []

    def handler(request: httpx.Request) -> httpx.Response:
        called.append(str(request.url))
        if request.url.host == "data.sec.gov":
            return httpx.Response(200, json=sec_recent())
        if request.url.host == "www.sec.gov":
            return httpx.Response(200, content=text)
        raise AssertionError("Unexpected host")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        params = {
            "db": db, "ciks": [1045810],
            "since": date(2026, 4, 1), "until": date(2026, 11, 18),
            "user_agent": "TrisResearch email@example.com",
            "client": client,
        }
        preview = collect_sec_filings(**params)
        assert preview["candidateTotal"] == 2
        assert preview["importedTotal"] == 0
        assert len(called) == 1
        result = collect_sec_filings(**params, import_files=True)
        assert result["importedTotal"] == 2
        again = collect_sec_filings(**params, import_files=True)
        assert again["importedTotal"] == 0
    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT a.source, a.title, a.published_at, a.detected_at, "
            "c.payload, c.sha256 FROM articles a JOIN captures c "
            "ON c.id=a.capture_id ORDER BY a.id"
        ).fetchall()
    assert len(rows) == 2
    assert rows[0][0] == "sec_8k_1045810"
    assert "2.02" in rows[0][1]
    assert rows[0][2] == "2026-08-19T18:32:10+00:00"
    assert datetime.fromisoformat(rows[0][3]).tzinfo is not None
    assert rows[0][4] == text
    import hashlib

    assert hashlib.sha256(rows[0][4]).hexdigest() == rows[0][5]
    event = {
        "source": rows[0][0], "title": rows[0][1],
        "summary": source_body(rows[0][4]),
        "sourceUrl": result["issuers"][0]["candidates"][0]["sourceUrl"],
    }
    policy = research_candidate(event)
    assert policy["eligible"] is True
    assert policy["eventCategory"] == "EARNINGS_OR_GUIDANCE"
    assert result["SECReceiptTimeNotFirstPublicRelease"] is True


def test_sec_issuer_still_requires_provenance_and_will_not_create_unknown_db(
    tmp_path,
):
    with pytest.raises(ValueError, match="EXISTING_ARTICLE_DATABASE_REQUIRED"):
        collect_sec_filings(
            db=tmp_path / "absent.sqlite3", ciks=[1045810],
            since=date(2026, 5, 1), until=date(2026, 11, 18),
            user_agent="Research email@example.com", import_files=True,
        )
    with pytest.raises(ValueError, match="SEC_IDENTIFYING_USER_AGENT_REQUIRED"):
        collect_sec_filings(
            db=tmp_path / "absent.sqlite3", ciks=[1045810],
            since=date(2026, 5, 1), until=date(2026, 11, 18),
            user_agent="TrisMarketEventIntelligence", import_files=False,
        )
