"""Stage 2-J: production v3 only for freshly collected, strict unique stories."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from tris_market_event_intelligence.collector import NewsStore, parse_rss
from tris_market_event_intelligence.event_store import EventProcessor
from tris_market_event_intelligence.integrated_pipeline import run_integrated_pass
from tris_market_event_intelligence.listing_research_workflow import (
    acquire_and_reconcile_listings,
)

RSS = (
    "<rss><channel>"
    "<item><guid>meta-1</guid><title>Italy fines Meta €805m over users' photos</title>"
    "<link>https://example.org/meta?utm_source=feed</link></item>"
    "<item><guid>ms-1</guid>"
    "<title>Microsoft says services fully restored after day-long global outage</title>"
    "<link>https://example.org/ms</link></item>"
    "</channel></rss>"
).encode()


def bundle_in(tmp_path: Path) -> Path:
    csv = tmp_path / "krx.csv"
    csv.write_text(
        "market,symbol,company_name,security_type\n"
        "XKRX,005930,삼성전자,COMMON\n", encoding="utf-8",
    )
    result = tmp_path / "issuer-source"
    acquire_and_reconcile_listings(
        result, krx_reviewed=csv, now=datetime.now(UTC) - timedelta(minutes=10),
    )
    return result


def make_client() -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, content=RSS),
    ))


def test_v3_fresh_only_ignores_historical_v2_and_dedups_two_feeds(
    tmp_path: Path,
) -> None:
    bundle = bundle_in(tmp_path)
    report_root = tmp_path / "report-root"
    report_root.mkdir()
    db = tmp_path / "news.sqlite3"
    store = NewsStore(db)
    store.initialize()
    prior = (
        "<rss><channel><item><guid>old</guid>"
        "<title>Cloud service outage disrupts users</title>"
        "<link>https://example.org/old</link></item></channel></rss>"
    ).encode()
    old_at = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    store.save("bbc_world", "https://example.org/feed", prior,
               parse_rss("bbc_world", prior, old_at),
               old_at, None, None)
    baseline = EventProcessor(db).process_pending()
    assert baseline.events == 1
    with sqlite3.connect(db) as connection:
        old_rows = connection.execute(
            "SELECT id, article_id, extractor_version FROM economic_events"
        ).fetchall()
        prior_checkpoints = connection.execute(
            "SELECT article_id, extractor_version FROM extraction_runs"
        ).fetchall()
    with make_client() as client:
        result = run_integrated_pass(
            db, selected_sources=("bbc_technology", "bbc_world"),
            client=client, listing_bundle=bundle,
            review_output_root=report_root, enable_v3_for_new_articles=True,
        )
    assert result["status"] == "healthy"
    assert result["event_extractor_version"] == "multi-v3"
    assert result["v3_fresh_only"] is True
    assert result["collection"]["new_articles"] == 4
    assert result["event_extraction"]["processed"] == 4
    assert result["event_extraction"]["events"] == 2
    assert result["event_extraction"]["verified"] == 0
    assert result["model_invoked"] is False
    with sqlite3.connect(db) as connection:
        persisted = connection.execute(
            "SELECT id, article_id, extractor_version, status, classification "
            "FROM economic_events ORDER BY id"
        ).fetchall()
        checkpoints = connection.execute(
            "SELECT article_id, extractor_version FROM extraction_runs "
            "ORDER BY article_id, extractor_version"
        ).fetchall()
    assert [row[:3] for row in persisted[:1]] == old_rows
    assert checkpoints[0] == prior_checkpoints[0]
    assert len(persisted) == 3
    assert [row[2:] for row in persisted[1:]] == [
        ("multi-v3", "UNVERIFIED", "REGULATION"),
        ("multi-v3", "UNVERIFIED", "SERVICE_INCIDENT"),
    ]
    assert len(checkpoints) == 5
    assert all(version == "multi-v3" for _, version in checkpoints[1:])
    with make_client() as client:
        repeated = run_integrated_pass(
            db, selected_sources=("bbc_technology", "bbc_world"),
            client=client, listing_bundle=bundle,
            review_output_root=report_root, enable_v3_for_new_articles=True,
        )
    assert repeated["collection"]["new_articles"] == 0
    assert repeated["event_extraction"] == {
        "processed": 0, "events": 0, "verified": 0,
    }
    with sqlite3.connect(db) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM economic_events"
        ).fetchone()[0] == 3


def test_v3_cannot_run_without_explicit_ticker_guard(
    tmp_path: Path,
) -> None:
    db = tmp_path / "news.sqlite3"
    with pytest.raises(ValueError, match="pinned listings"):
        run_integrated_pass(db, enable_v3_for_new_articles=True)
    assert not db.exists()


def test_v3_is_not_backfill_for_existing_post_snapshot_news(
    tmp_path: Path,
) -> None:
    bundle = bundle_in(tmp_path)
    folder = tmp_path / "reports"
    folder.mkdir()
    db = tmp_path / "news.sqlite3"
    store = NewsStore(db)
    store.initialize()
    seen = datetime.now(UTC).isoformat()
    store.save(
        "bbc_world", "https://example.org/feed", RSS,
        parse_rss("bbc_world", RSS, seen), seen, None, None,
    )
    with make_client() as client:
        result = run_integrated_pass(
            db, selected_sources=("bbc_world",), client=client,
            listing_bundle=bundle, review_output_root=folder,
            enable_v3_for_new_articles=True,
        )
    assert result["status"] == "healthy"
    assert result["collection"]["new_articles"] == 0
    assert result["event_extraction"]["processed"] == 0
    with sqlite3.connect(db) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM economic_events"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM extraction_runs"
        ).fetchone()[0] == 0
