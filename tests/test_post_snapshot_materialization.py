"""Offline end-to-end: recorded RSS -> post-capture events -> ticker candidates.

No actual network, live DB or automatic model calls.
"""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tris_market_event_intelligence.collector import NewsStore, parse_rss
from tris_market_event_intelligence.listing_research_workflow import (
    acquire_and_reconcile_listings,
)
from tris_market_event_intelligence.post_snapshot_materialization import (
    materialize_post_snapshot_events,
)

CAPTURE_TIME = datetime(2026, 11, 15, 0, 0, tzinfo=UTC)
CUTOFF = datetime(2026, 11, 15, 4, 0, tzinfo=UTC)


def original_snapshot(tmp_path: Path) -> Path:
    actual = tmp_path / "krx-reviewed.csv"
    actual.write_text(
        "market,symbol,company_name,security_type\n"
        "XKRX,005930,삼성전자,COMMON\n", encoding="utf-8",
    )
    bundle = tmp_path / "captured"
    acquire_and_reconcile_listings(bundle, krx_reviewed=actual, now=CAPTURE_TIME)
    return bundle


def captured_article(store: NewsStore, *, guid: str, time: str, url: str,
                     title: str) -> None:
    rss = (
        f"<rss><channel><item><guid>{guid}</guid>"
        f"<title>{title}</title><link>{url}</link>"
        "</item></channel></rss>"
    ).encode()
    article = parse_rss("bbc_business", rss, time)
    store.save("bbc_business", "https://example.org/feed", rss, article, time, None, None)


def test_post_capture_real_event_creation_then_review_candidate(
    tmp_path: Path,
) -> None:
    bundle = original_snapshot(tmp_path)
    db = tmp_path / "news.sqlite3"
    store = NewsStore(db)
    store.initialize()
    captured_article(
        store, guid="old", time="2026-11-14T23:00:00+00:00",
        url="https://example.org/old", title="삼성전자 서비스 장애",
    )
    captured_article(
        store, guid="new", time="2026-11-15T01:00:00+00:00",
        url="https://example.org/new", title="삼성전자 서비스 장애",
    )
    captured_article(
        store, guid="unclassified", time="2026-11-15T02:00:00+00:00",
        url="https://example.org/no-event", title="Sunny day in Paris",
    )
    report = materialize_post_snapshot_events(
        bundle, db, tmp_path / "analysis", as_of=CUTOFF,
    )
    assert report["status"] == "EXTRACTED_AND_VALIDATED"
    assert report["event_extraction"] == {
        "processed": 2, "events": 1, "verified": 0,
    }
    assert report["post_snapshot_events_inspected"] == 1
    assert report["ticker_candidates"] == 1
    assert report["model_invoked"] is False
    assert report["collector_invoked"] is False
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM economic_events").fetchone()[0] == 1
        assert connection.execute(
            "SELECT a.detected_at FROM economic_events e "
            "JOIN articles a ON a.id=e.article_id"
        ).fetchone()[0] == "2026-11-15T01:00:00+00:00"
        assert connection.execute("SELECT COUNT(*) FROM extraction_runs").fetchone()[0] == 2
    again = materialize_post_snapshot_events(
        bundle, db, tmp_path / "rerun", as_of=CUTOFF,
    )
    assert again["event_extraction"]["processed"] == 0
    assert again["ticker_candidates"] == 1


def test_bounded_extraction_never_drains_backlog_without_request(tmp_path: Path) -> None:
    bundle = original_snapshot(tmp_path)
    db = tmp_path / "news.sqlite3"
    store = NewsStore(db)
    store.initialize()
    for i in range(3):
        captured_article(
            store, guid=f"new-{i}", time="2026-11-15T01:00:00+00:00",
            url=f"https://example.org/{i}", title="삼성전자 서비스 장애",
        )
    first = materialize_post_snapshot_events(
        bundle, db, tmp_path / "first", batch_size=1, as_of=CUTOFF,
    )
    assert first["event_extraction"]["processed"] == 1
    assert first["more_pending_articles_not_excluded"] is True
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM economic_events").fetchone()[0] == 1
    second = materialize_post_snapshot_events(
        bundle, db, tmp_path / "second", batch_size=1, as_of=CUTOFF,
    )
    assert second["event_extraction"]["processed"] == 1
    assert second["post_snapshot_events_inspected"] == 2


def test_no_eligible_new_articles_creates_no_events(tmp_path: Path) -> None:
    bundle = original_snapshot(tmp_path)
    db = tmp_path / "news.sqlite3"
    store = NewsStore(db)
    store.initialize()
    captured_article(
        store, guid="old", time="2026-11-14T23:00:00+00:00",
        url="https://example.org/old", title="삼성전자 서비스 장애",
    )
    report = materialize_post_snapshot_events(
        bundle, db, tmp_path / "empty", as_of=CUTOFF,
    )
    assert report["event_extraction"]["processed"] == 0
    assert report["post_snapshot_events_inspected"] == 0
    with sqlite3.connect(db) as connection:
        assert connection.execute("SELECT COUNT(*) FROM economic_events").fetchone()[0] == 0


def test_invalid_paths_or_limits_leave_database_untouched(tmp_path: Path) -> None:
    bundle = original_snapshot(tmp_path)
    db = tmp_path / "news.sqlite3"
    store = NewsStore(db)
    store.initialize()
    original = db.read_bytes()
    with pytest.raises(ValueError, match="batch_size"):
        materialize_post_snapshot_events(bundle, db, tmp_path / "bad", batch_size=1000)
    with pytest.raises(ValueError, match="new directory"):
        materialize_post_snapshot_events(bundle, db, tmp_path, as_of=CUTOFF)
    assert not (tmp_path / "bad").exists()
    assert db.read_bytes() == original
