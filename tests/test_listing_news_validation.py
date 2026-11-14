"""Actual captured source and stored news validation, with no external APIs."""
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tris_market_event_intelligence.listing_news_validation import (
    validate_existing_listing_news,
)
from tris_market_event_intelligence.listing_research_workflow import (
    acquire_and_reconcile_listings,
)

SNAPSHOT_TIME = datetime(2026, 11, 15, 0, 0, tzinfo=UTC)
EVALUATED = datetime(2026, 11, 15, 4, 0, tzinfo=UTC)
KRX = (
    "market,symbol,company_name,security_type\n"
    "XKRX,005930,삼성전자,COMMON\n"
    "XKRX,000660,SK하이닉스,COMMON\n"
)


def capture(tmp_path: Path) -> Path:
    raw = tmp_path / "krx-source.csv"
    raw.write_text(KRX, encoding="utf-8")
    bundle = tmp_path / "observed-offline"
    acquire_and_reconcile_listings(
        bundle, krx_reviewed=raw, now=SNAPSHOT_TIME,
    )
    return bundle


def events(path: Path, records: list[tuple[str, str]]) -> None:
    with sqlite3.connect(path) as db:
        db.executescript(
            """CREATE TABLE articles (
                id INTEGER PRIMARY KEY, title TEXT, summary TEXT, detected_at TEXT
            );
            CREATE TABLE economic_events (
                id INTEGER PRIMARY KEY, article_id INTEGER, status TEXT,
                source TEXT, source_url TEXT
            );"""
        )
        for index, (headline, seen) in enumerate(records, start=1):
            db.execute("INSERT INTO articles VALUES (?,?,'',?)", (index, headline, seen))
            db.execute(
                "INSERT INTO economic_events VALUES (?,?,'VERIFIED','test',?)",
                (index, index, f"https://example.org/{index}"),
            )


def test_reuse_capture_reports_old_and_new_events_and_review_status(tmp_path: Path) -> None:
    bundle = capture(tmp_path)
    db = tmp_path / "news.sqlite3"
    events(db, [
        ("삼성전자 announces earnings", "2026-11-14T23:00:00+00:00"),
        ("삼성전자 announces earnings", "2026-11-15T02:00:00+00:00"),
        ("Generic macro headline", "2026-11-15T03:00:00+00:00"),
        ("SK하이닉스 issues statement", "2026-11-15T05:00:00+00:00"),
    ])
    before = db.read_bytes()
    result = validate_existing_listing_news(
        bundle, db, tmp_path / "analysis", as_of=EVALUATED,
    )
    assert result["status"] == "REVIEW_CANDIDATES_PRESENT"
    assert result["listing_issuer_rows"] == 2
    assert result["events_inspected"] == 4
    assert result["pre_observation_news_events"] == 1
    assert result["post_observation_news_events"] == 3
    assert result["candidate_count"] == 1
    assert result["pending_human_review"] == 1
    assert result["has_more_events"] is False
    assert result["status_counts"] == {
        "FUTURE_NEWS_TIMESTAMP": 1,
        "FUTURE_REGISTRY_EVIDENCE": 1,
        "NO_VERIFIED_NAME_MATCH": 1,
        "PROPOSED": 1,
    }
    page = json.loads((tmp_path / "analysis" / "proposals-page-001.json").read_text())
    assert [item["status"] for item in page["items"]] == [
        "FUTURE_REGISTRY_EVIDENCE", "PROPOSED",
        "NO_VERIFIED_NAME_MATCH", "FUTURE_NEWS_TIMESTAMP",
    ]
    assert page["items"][1]["proposals"][0]["symbol"] == "005930"
    assert result["candidate_samples"][0]["candidates"] == [
        {"market": "XKRX", "symbol": "005930"}
    ]
    assert result["point_in_time_training_eligible"] is False
    assert result["database_modified"] is False
    assert db.read_bytes() == before
    assert (bundle / "krx-reviewed.csv").exists()
    assert (tmp_path / "analysis" / "summary.json").is_file()


def test_large_news_scan_pages_with_explicit_incomplete_boundary(tmp_path: Path) -> None:
    bundle = capture(tmp_path)
    db = tmp_path / "news.sqlite3"
    records = [("삼성전자 issue", "2026-11-15T02:00:00+00:00")] * 503
    events(db, records)
    one = validate_existing_listing_news(
        bundle, db, tmp_path / "partial", max_pages=1, as_of=EVALUATED,
    )
    assert one["events_inspected"] == 500
    assert one["candidate_count"] == 500
    assert one["has_more_events"] is True
    assert one["resume_after_event_id"] == 500
    full = validate_existing_listing_news(
        bundle, db, tmp_path / "full", max_pages=2, as_of=EVALUATED,
    )
    assert full["events_inspected"] == 503
    assert full["candidate_count"] == 503
    assert full["has_more_events"] is False
    assert len(full["pages"]) == 2


def test_registry_cannot_diverge_from_hash_pinned_sources(tmp_path: Path) -> None:
    bundle = capture(tmp_path)
    db = tmp_path / "news.sqlite3"
    events(db, [("삼성전자", "2026-11-15T02:00:00+00:00")])
    registry = bundle / "review-required-registry.csv"
    registry.write_text(
        registry.read_text().replace("삼성전자", "가짜회사"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Registry differs"):
        validate_existing_listing_news(
            bundle, db, tmp_path / "analysis", as_of=EVALUATED,
        )
    assert not (tmp_path / "analysis").exists()


def test_corrupt_original_snapshot_does_not_create_analysis(tmp_path: Path) -> None:
    bundle = capture(tmp_path)
    (bundle / "krx-reviewed.csv").write_text("tampered")
    db = tmp_path / "news.sqlite3"
    events(db, [("삼성전자", "2026-11-15T02:00:00+00:00")])
    with pytest.raises(ValueError, match="SHA-256"):
        validate_existing_listing_news(
            bundle, db, tmp_path / "analysis", as_of=EVALUATED,
        )
    assert not (tmp_path / "analysis").exists()


def test_missing_news_and_existing_output_refused(tmp_path: Path) -> None:
    bundle = capture(tmp_path)
    with pytest.raises(ValueError, match="news database"):
        validate_existing_listing_news(
            bundle, tmp_path / "missing.db", tmp_path / "new",
            as_of=EVALUATED,
        )
    assert not (tmp_path / "missing.db").exists()
    db = tmp_path / "news.sqlite3"
    events(db, [])
    output = tmp_path / "keep"
    output.mkdir()
    (output / "personal.txt").write_text("leave untouched")
    with pytest.raises(ValueError, match="new folder"):
        validate_existing_listing_news(bundle, db, output, as_of=EVALUATED)
    assert (output / "personal.txt").read_text() == "leave untouched"
