"""Stage 2-K uses immutable activation, not the most recent manual run."""
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

STORY = (
    "<rss><channel>"
    "<item><guid>meta</guid>"
    "<title>Italy fines Meta €805m over accessing users' photos</title>"
    "<link>https://example.org/meta?utm_source=rss</link></item>"
    "<item><guid>microsoft</guid>"
    "<title>Microsoft says services fully restored after day-long global outage</title>"
    "<link>https://example.org/ms</link></item></channel></rss>"
).encode()


def setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    reviewed = tmp_path / "review.csv"
    reviewed.write_text(
        "market,symbol,company_name,security_type\n"
        "XKRX,005930,삼성전자,COMMON\n", encoding="utf-8",
    )
    bundle = tmp_path / "listing"
    acquire_and_reconcile_listings(
        bundle, krx_reviewed=reviewed,
        now=datetime.now(UTC) - timedelta(minutes=30),
    )
    output = tmp_path / "reports"
    output.mkdir()
    return bundle, output, tmp_path / "news.sqlite3"


def client_for(payload: bytes) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, content=payload),
    ))


def test_scheduler_first_article_is_processed_once_after_pinned_cutover(
    tmp_path: Path,
) -> None:
    bundle, output, db = setup(tmp_path)
    store = NewsStore(db)
    store.initialize()
    old = (
        b"<rss><channel><item><guid>old</guid>"
        b"<title>Cloud service outage disrupts users</title>"
        b"<link>https://example.org/old</link></item></channel></rss>"
    )
    old_time = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    store.save(
        "bbc_world", "https://example.org/feed", old,
        parse_rss("bbc_world", old, old_time), old_time, None, None,
    )
    # First opt-in pins a fence but cannot process preactivation backlog.
    empty = b"<rss><channel></channel></rss>"
    with client_for(empty) as client:
        first = run_integrated_pass(
            db, selected_sources=("bbc_world",), client=client,
            listing_bundle=bundle, review_output_root=output,
            enable_v3_cutover_catchup=True,
        )
    assert first["status"] == "healthy"
    assert first["v3_cutover_created"] is True
    assert first["v3_cutover_catchup"] is True
    assert first["v3_cutover_article_floor"] == 1
    assert first["event_extraction"]["processed"] == 0
    marker = first["v3_cutover_activated_at"]
    # Existing 15-min collector inserts records BEFORE next manual v3 pass.
    receipt = datetime.now(UTC).isoformat()
    saved = store.save(
        "bbc_world", "https://example.org/feed", STORY,
        parse_rss("bbc_world", STORY, receipt), receipt, None, None,
    )
    assert saved == 2
    with client_for(STORY) as client:
        second = run_integrated_pass(
            db, selected_sources=("bbc_world",), client=client,
            listing_bundle=bundle, review_output_root=output,
            enable_v3_cutover_catchup=True,
        )
    assert second["status"] == "healthy"
    assert second["collection"]["new_articles"] == 0
    assert second["v3_cutover_created"] is False
    assert second["v3_cutover_activated_at"] == marker
    assert second["v3_cutover_article_floor"] == 1
    assert second["event_extractor_version"] == "multi-v3"
    assert second["event_extraction"] == {
        "processed": 2, "events": 2, "verified": 0,
    }
    assert second["model_invoked"] is False
    with sqlite3.connect(db) as connection:
        events = connection.execute(
            "SELECT article_id, extractor_version, status FROM economic_events ORDER BY id"
        ).fetchall()
        checkpoints = connection.execute(
            "SELECT article_id, extractor_version FROM extraction_runs ORDER BY article_id"
        ).fetchall()
        activation_rows = connection.execute(
            "SELECT activated_at, article_id_floor FROM event_extractor_rollout"
        ).fetchall()
    assert events == [(2, "multi-v3", "UNVERIFIED"), (3, "multi-v3", "UNVERIFIED")]
    assert checkpoints == [(2, "multi-v3"), (3, "multi-v3")]
    assert activation_rows == [(marker, 1)]
    with client_for(STORY) as client:
        third = run_integrated_pass(
            db, selected_sources=("bbc_world",), client=client,
            listing_bundle=bundle, review_output_root=output,
            enable_v3_cutover_catchup=True,
        )
    assert third["event_extraction"]["processed"] == 0
    assert third["event_extraction"]["events"] == 0


def test_v2_checkpoint_after_activation_is_not_replayed_by_v3(
    tmp_path: Path,
) -> None:
    bundle, output, db = setup(tmp_path)
    empty = b"<rss><channel></channel></rss>"
    with client_for(empty) as client:
        run_integrated_pass(
            db, selected_sources=("bbc_world",), client=client,
            listing_bundle=bundle, review_output_root=output,
            enable_v3_cutover_catchup=True,
        )
    store = NewsStore(db)
    now = datetime.now(UTC).isoformat()
    store.save(
        "bbc_world", "https://example.org/feed", STORY,
        parse_rss("bbc_world", STORY, now), now, None, None,
    )
    old_processor = EventProcessor(db).process_pending()
    assert old_processor.processed == 2
    assert old_processor.events == 0
    with client_for(STORY) as client:
        result = run_integrated_pass(
            db, selected_sources=("bbc_world",), client=client,
            listing_bundle=bundle, review_output_root=output,
            enable_v3_cutover_catchup=True,
        )
    assert result["event_extraction"]["processed"] == 0
    assert result["event_extraction"]["events"] == 0
    with sqlite3.connect(db) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM economic_events"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM extraction_runs WHERE extractor_version='multi-v2'"
        ).fetchone()[0] == 2


def test_v3_cutover_guard_prevents_unapproved_db_changes(tmp_path: Path) -> None:
    db = tmp_path / "news.sqlite3"
    with pytest.raises(ValueError, match="pinned listings"):
        run_integrated_pass(db, enable_v3_cutover_catchup=True)
    with pytest.raises(ValueError, match="only one v3"):
        run_integrated_pass(
            db, enable_v3_cutover_catchup=True,
            enable_v3_for_new_articles=True,
        )
    assert not db.exists()
