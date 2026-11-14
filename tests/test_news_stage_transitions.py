"""Read-only stage evidence: polling is not articles, and articles are not events."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.news_stage_transitions import (
    inspect_news_stage_transitions,
)

FIRST_SEEN = datetime(2026, 11, 14, 15, 54, tzinfo=UTC)
CUTOFF = datetime(2026, 11, 14, 20, 0, tzinfo=UTC)


def make_news(path: Path) -> None:
    with sqlite3.connect(path) as db:
        db.executescript(
            """CREATE TABLE articles(
                id INTEGER PRIMARY KEY, source TEXT NOT NULL, url TEXT,
                detected_at TEXT NOT NULL
            );
            CREATE TABLE extraction_runs(
                article_id INTEGER, extractor_version TEXT, processed_at TEXT
            );
            CREATE TABLE economic_events(
                id INTEGER PRIMARY KEY, article_id INTEGER, status TEXT,
                processed_at TEXT
            );
            """
        )
        db.executemany("INSERT INTO articles VALUES (?,?,?,?)", [
            (1, "bbc_business", "https://example.org/old",
             "2026-11-14T15:00:00+00:00"),
            (2, "bbc_technology", "https://example.org/new-1",
             "2026-11-14T16:00:00+00:00"),
            (3, "bbc_business", "https://example.org/new-2",
             "2026-11-14T16:01:00+00:00"),
            (4, "bbc_business", "https://example.org/new-3",
             "2026-11-14T16:02:00+00:00"),
            (5, "bbc_business", "https://example.org/future",
             "2026-11-14T21:00:00+00:00"),
        ])
        db.executemany("INSERT INTO extraction_runs VALUES (?,?,?)", [
            (1, "multi-v2", "2026-11-14T15:01:00+00:00"),
            (3, "multi-v2", "2026-11-14T16:02:00+00:00"),
            (4, "multi-v2", "2026-11-14T16:03:00+00:00"),
        ])
        db.execute(
            "INSERT INTO economic_events VALUES (10,4,'UNVERIFIED',?)",
            ("2026-11-14T16:04:00+00:00",),
        )


def test_readonly_funnel_distinguishes_pending_and_unclassified(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    make_news(path)
    before = path.read_bytes()
    result = inspect_news_stage_transitions(
        path, snapshot_observed_at=FIRST_SEEN, as_of=CUTOFF,
    )
    assert result["status"] == "OBSERVED_READ_ONLY"
    assert result["new_articles"] == 3
    assert result["pending_extraction_articles"] == 1
    assert result["extraction_checkpointed_articles"] == 2
    assert result["processed_without_classified_event"] == 1
    assert result["articles_with_events"] == 1
    assert result["event_rows"] == 1
    assert result["event_processing_delay_seconds"] == {
        "count": 1, "min": 120.0, "max": 120.0,
    }
    assert result["by_source"]["bbc_business"]["articles"] == 2
    assert result["by_source"]["bbc_technology"]["pending_extraction"] == 1
    assert {sample["stage"] for sample in result["unresolved_article_samples"]} == {
        "AWAITING_EVENT_EXTRACTION", "EXTRACTED_NO_CLASSIFIED_EVENT",
    }
    assert result["ticker_candidates_automatically_approved"] == 0
    assert result["database_modified"] is False
    assert path.read_bytes() == before


def test_zero_articles_is_not_reported_as_collection_failure(tmp_path: Path) -> None:
    path = tmp_path / "empty.db"
    make_news(path)
    result = inspect_news_stage_transitions(
        path,
        snapshot_observed_at=datetime(2026, 11, 16, tzinfo=UTC),
        as_of=datetime(2026, 11, 17, tzinfo=UTC),
    )
    assert result["status"] == "OBSERVED_READ_ONLY"
    assert result["new_articles"] == result["event_rows"] == 0
    assert result["by_source"] == {}
    assert result["event_processing_delay_seconds"]["count"] == 0


def test_legacy_schema_and_missing_database_report_safely(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE articles(id INTEGER, detected_at TEXT)")
    result = inspect_news_stage_transitions(
        path, snapshot_observed_at=FIRST_SEEN, as_of=CUTOFF,
    )
    assert result["status"] == "UNAVAILABLE_SCHEMA"
    assert "economic_events" in result["missing_tables"]
    assert result["database_modified"] is False
    missing = tmp_path / "missing.db"
    try:
        inspect_news_stage_transitions(
            missing, snapshot_observed_at=FIRST_SEEN, as_of=CUTOFF,
        )
    except ValueError as exc:
        assert "Existing nonsymlink" in str(exc)
    else:
        raise AssertionError("Expected missing database to be rejected")
    assert not missing.exists()
