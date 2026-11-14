"""Offline classification and cross-source URL grouping with preserved evidence.

All news text and URLs in tests are synthetic, not representations of the
five specific real-world articles in the owner's private Mac SQLite.
"""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.news_classification_evidence import (
    inspect_unclassified_news,
)

FIRST = datetime(2026, 11, 14, 15, 54, tzinfo=UTC)
END = datetime(2026, 11, 14, 18, 0, tzinfo=UTC)


def create_test_db(path: Path) -> None:
    with sqlite3.connect(path) as db:
        db.executescript(
            """CREATE TABLE articles(
                id INTEGER PRIMARY KEY, source TEXT, url TEXT, title TEXT,
                summary TEXT, detected_at TEXT
            );
            CREATE TABLE extraction_runs(
                article_id INTEGER, extractor_version TEXT, processed_at TEXT
            );
            CREATE TABLE economic_events(
                id INTEGER PRIMARY KEY, article_id INTEGER
            );"""
        )
        db.executemany(
            "INSERT INTO articles VALUES(?,?,?,?,?,?)",
            [
                (1, "bbc_world", "https://example.org/old",
                 "Cloud service outage disrupts users", None,
                 "2026-11-14T15:00:00+00:00"),
                (2, "bbc_world", "https://example.org/same?utm_source=world",
                 "Local weather on Sunday", None,
                 "2026-11-14T16:00:00+00:00"),
                (3, "bbc_technology", "https://example.org/same?utm_source=tech",
                 "Local weather on Sunday", None,
                 "2026-11-14T16:01:00+00:00"),
                (4, "bbc_business", "https://example.org/cfo",
                 "CEO resignation announced", None,
                 "2026-11-14T16:02:00+00:00"),
                (5, "bbc_technology", "https://example.org/outage",
                 "Cloud service outage disrupts users", None,
                 "2026-11-14T16:03:00+00:00"),
                (6, "bbc_world", "https://example.org/unprocessed",
                 "Ordinary neighborhood story", None,
                 "2026-11-14T16:04:00+00:00"),
                (7, "bbc_world", "https://example.org/future",
                 "Cloud service outage disrupts users", None,
                 "2026-11-14T20:00:00+00:00"),
            ],
        )
        db.executemany(
            "INSERT INTO extraction_runs VALUES(?,?,?)",
            [
                (2, "multi-v2", "2026-11-14T16:05:00+00:00"),
                (3, "multi-v2", "2026-11-14T16:05:00+00:00"),
                (4, "multi-v2", "2026-11-14T16:05:00+00:00"),
                (5, "multi-v2", "2026-11-14T16:05:00+00:00"),
            ],
        )
        db.execute("INSERT INTO economic_events VALUES(11,5)")


def test_actual_article_stage_reasons_and_same_url_across_sources(
    tmp_path: Path,
) -> None:
    db = tmp_path / "news.sqlite3"
    create_test_db(db)
    original = db.read_bytes()
    report = inspect_unclassified_news(db, observed_at=FIRST, as_of=END)
    assert report["status"] == "OBSERVED_READ_ONLY"
    assert report["articles_inspected"] == 5
    assert report["reason_counts"]["NO_SUPPORTED_EVENT_CUE"] == 2
    assert report["reason_counts"]["CLAIM_ONLY_NO_EVENT"] == 1
    assert report["reason_counts"]["HAS_STORED_EVENT"] == 1
    assert report["reason_counts"]["AWAITING_EXTRACTION"] == 1
    assert report["cross_source_canonical_url_groups"] == 1
    assert report["unique_canonical_url_groups"] == 4
    assert report["automatic_ticker_approvals"] == 0
    assert report["cross_source_events_suppressed"] == 0
    samples = {item["article_id"]: item for item in report["article_samples"]}
    assert samples[2]["cross_source_same_url_article_ids"] == [2, 3]
    assert samples[3]["cross_source_same_url_article_ids"] == [2, 3]
    assert samples[4]["current_rule_classification"] is None
    assert samples[5]["current_rule_classification"] == "SERVICE_INCIDENT"
    assert report["database_modified"] is False
    assert report["network_requests"] == 0
    assert db.read_bytes() == original


def test_empty_window_does_not_imply_missing_news_or_failure(tmp_path: Path) -> None:
    db = tmp_path / "news.sqlite3"
    create_test_db(db)
    report = inspect_unclassified_news(
        db,
        observed_at=datetime(2026, 11, 16, tzinfo=UTC),
        as_of=datetime(2026, 11, 17, tzinfo=UTC),
    )
    assert report["articles_inspected"] == 0
    assert report["reason_counts"] == {}
    assert report["cross_source_canonical_url_groups"] == 0


def test_incomplete_legacy_db_does_not_claim_clean_evidence(tmp_path: Path) -> None:
    db = tmp_path / "old.sqlite3"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE articles(id INTEGER PRIMARY KEY)")
    report = inspect_unclassified_news(db, observed_at=FIRST, as_of=END)
    assert report["status"] == "UNAVAILABLE_SCHEMA"
    assert "extraction_runs" in report["missing_tables"]
    assert report["database_modified"] is False
