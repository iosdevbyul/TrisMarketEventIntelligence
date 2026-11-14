"""Read-only evidence separates raw story duplicates from event versions."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.news_pipeline_evidence import inspect_news_pipeline

OBSERVED = datetime(2026, 11, 14, 15, 54, tzinfo=UTC)
NOW = datetime(2026, 11, 15, 0, 0, tzinfo=UTC)


def test_audit_existing_duplicate_provenance_without_modification(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    with sqlite3.connect(path) as db:
        db.executescript(
            """CREATE TABLE articles (
              id INTEGER PRIMARY KEY, source TEXT, url TEXT, title TEXT,
              detected_at TEXT
            );
            CREATE TABLE economic_events (
              id INTEGER PRIMARY KEY, article_id INTEGER,
              source TEXT, source_url TEXT
            );
            CREATE TABLE fetch_attempts (
              id INTEGER PRIMARY KEY, source TEXT, status TEXT,
              inserted INTEGER, completed_at TEXT
            );"""
        )
        for row in [
            (1, "bbc_business", "https://example.org/story?utm_source=rss",
             "Microsoft reports outage", "2026-11-13T20:11:59+00:00"),
            (2, "bbc_business", "https://example.org/story?utm_source=rss",
             "Microsoft reports outage", "2026-11-13T20:12:00+00:00"),
            (3, "bbc_world", "https://example.org/story?utm_source=other",
             "Microsoft reports outage", "2026-11-15T00:00:00+09:00"),
            (4, "bbc_world", "https://example.org/new",
             "New market results", "2026-11-14T18:00:00+00:00"),
        ]:
            db.execute("INSERT INTO articles VALUES (?,?,?,?,?)", row)
        for row in [
            (1, 1, "bbc_business", "https://example.org/story?utm_source=rss"),
            (2, 2, "bbc_business", "https://example.org/story?utm_source=rss"),
            (3, 3, "bbc_world", "https://example.org/story?utm_source=other"),
            (4, 4, "bbc_world", "https://example.org/new"),
            (5, 1, "bbc_business", "https://example.org/story?utm_source=rss"),
        ]:
            db.execute("INSERT INTO economic_events VALUES (?,?,?,?)", row)
        db.execute(
            "INSERT INTO fetch_attempts VALUES (1,'bbc_world','ok',1,?)",
            ("2026-11-14T19:00:00+00:00",),
        )
    before = path.read_bytes()
    report = inspect_news_pipeline(path, snapshot_observed_at=OBSERVED, as_of=NOW)
    assert report["status"] == "OBSERVED_READ_ONLY"
    assert report["same_source_exact_url_duplicate_articles"] == 1
    assert report["canonical_url_multirecord_groups"] == 1
    assert report["multiple_events_per_article"] == 1
    assert report["articles_first_seen_after_snapshot"] == 2
    assert report["latest_fetch_attempts"]["bbc_world"]["status"] == "ok"
    assert report["news_collection_active_verified"] is False
    assert report["database_modified"] is False
    assert path.read_bytes() == before


def test_legacy_schema_reports_unknown_instead_of_zero_duplicates(tmp_path: Path) -> None:
    db = tmp_path / "small.db"
    with sqlite3.connect(db) as connection:
        connection.executescript(
            "CREATE TABLE articles (id INTEGER PRIMARY KEY, title TEXT);"
            "CREATE TABLE economic_events (id INTEGER PRIMARY KEY, article_id INTEGER);"
        )
    result = inspect_news_pipeline(db, snapshot_observed_at=OBSERVED, as_of=NOW)
    assert result["status"] == "UNAVAILABLE_SCHEMA"
    assert "same_source_exact_url_duplicate_articles" not in result
    assert result["database_modified"] is False
