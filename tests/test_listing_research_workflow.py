"""Opt-in live-source workflow uses injected HTTP fixtures, never actual network."""
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from tris_market_event_intelligence.listing_research_workflow import (
    US_DIRECTORY,
    acquire_and_reconcile_listings,
)

OBSERVED = datetime(2026, 11, 15, 0, 0, tzinfo=UTC)
NASDAQ = (
    "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|"
    "ETF|NextShares\n"
    "MSFT|Microsoft Corporation - Common Stock|Q|N|N|100|N|N\n"
    "ETFX|Some Fund|Q|N|N|100|Y|N\n"
    "File Creation Time: 1115202612:00|||||||\n"
).encode()
OTHER = (
    "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|"
    "NASDAQ Symbol\n"
    "ACME|Acme Corporation - Common Stock|N|ACME|N|100|N|ACME\n"
    "OTHER|Another Corp - Common Stock|P|OTHER|N|100|N|OTHER\n"
    "File Creation Time: 1115202612:00|||||||\n"
).encode()
KRX = (
    "market,symbol,company_name,security_type\n"
    "XKRX,005930,삼성전자,COMMON\n"
    "XKRX,999999,가상우선주,PREFERRED\n"
).encode()


def client(*, bad: bool = False) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) in {u for u, _ in US_DIRECTORY.values()}
        if bad and request.url.path.endswith("otherlisted.txt"):
            return httpx.Response(503)
        return httpx.Response(200, content=(
            NASDAQ if request.url.path.endswith("nasdaqlisted.txt") else OTHER
        ))

    return httpx.Client(transport=httpx.MockTransport(handler), trust_env=False)


def db_fixture(db: Path) -> None:
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """CREATE TABLE articles (
                 id INTEGER PRIMARY KEY, title TEXT, summary TEXT,
                 detected_at TEXT
               );
               CREATE TABLE economic_events (
                 id INTEGER PRIMARY KEY, article_id INTEGER, status TEXT,
                 source TEXT, source_url TEXT
               );"""
        )
        conn.execute(
            "INSERT INTO articles VALUES (1, ?, '', ?)",
            ("Microsoft Corporation announces results", "2026-11-14T20:00:00+00:00"),
        )
        conn.execute(
            "INSERT INTO economic_events VALUES (1, 1, 'VERIFIED', 'fixture', ?)",
            ("https://example.org/news/1",),
        )


def test_real_source_capture_reuses_existing_import_and_matcher(tmp_path: Path) -> None:
    news = tmp_path / "news.sqlite3"
    db_fixture(news)
    krx = tmp_path / "actual-krx-normalized.csv"
    krx.write_bytes(KRX)
    dest = tmp_path / "research"
    with client() as session:
        report = acquire_and_reconcile_listings(
            dest, fetch_us=True, krx_reviewed=krx, news_db=news,
            client=session, now=OBSERVED,
        )
    assert report["issuer_rows"] == 3
    assert report["ticker_candidates"] == 0
    assert report["events_inspected"] == 1
    assert report["status"] == "SNAPSHOTS_CAPTURED_REVIEW_REQUIRED"
    assert report["point_in_time_training_eligible"] is False
    assert report["database_modified"] is False
    assert report["model_invoked"] is False
    assert json.loads((dest / "manifest.json").read_text())["sources"][0]["kind"] == (
        "NASDAQ_DIRECTORY"
    )
    proposal = json.loads((dest / "proposals.json").read_text())
    assert proposal["items"][0]["status"] == "FUTURE_REGISTRY_EVIDENCE"
    assert (dest / "nasdaqlisted.txt").read_bytes() == NASDAQ
    assert (dest / "otherlisted.txt").read_bytes() == OTHER
    assert (dest / "krx-reviewed.csv").read_bytes() == KRX
    assert krx.read_bytes() == KRX
    assert len(report["source_reports"]) == 3


def test_source_failure_does_not_create_partial_bundle(tmp_path: Path) -> None:
    dest = tmp_path / "new"
    with client(bad=True) as session, pytest.raises(httpx.HTTPStatusError):
        acquire_and_reconcile_listings(
            dest, fetch_us=True, client=session, now=OBSERVED,
        )
    assert not dest.exists()


def test_invalid_krx_file_does_not_create_bundle(tmp_path: Path) -> None:
    bad = tmp_path / "invalid.csv"
    bad.write_text("wrong,columns\n1,2\n")
    with pytest.raises(ValueError):
        acquire_and_reconcile_listings(
            tmp_path / "new", krx_reviewed=bad, now=OBSERVED,
        )
    assert not (tmp_path / "new").exists()


def test_existing_artifacts_are_never_overwritten(tmp_path: Path) -> None:
    dest = tmp_path / "already"
    dest.mkdir()
    marker = dest / "personal-notes.txt"
    marker.write_text("keep")
    with client() as session, pytest.raises(ValueError, match="new directory"):
        acquire_and_reconcile_listings(
            dest, fetch_us=True, client=session, now=OBSERVED,
        )
    assert marker.read_text() == "keep"


def test_explicit_source_required_and_missing_news_db_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Select"):
        acquire_and_reconcile_listings(tmp_path / "missing")
    with client() as session, pytest.raises(ValueError, match="news DB"):
        acquire_and_reconcile_listings(
            tmp_path / "new", fetch_us=True,
            news_db=tmp_path / "none.sqlite3", client=session,
        )
    assert not (tmp_path / "new").exists()


def test_krx_only_operator_file_requires_no_network(tmp_path: Path) -> None:
    krx = tmp_path / "krx.csv"
    krx.write_bytes(KRX)
    report = acquire_and_reconcile_listings(
        tmp_path / "krx-only", krx_reviewed=krx, now=OBSERVED,
    )
    assert report["issuer_rows"] == 1
    assert report["news_matching_performed"] is False
    assert not (tmp_path / "krx-only" / "proposals.json").exists()
