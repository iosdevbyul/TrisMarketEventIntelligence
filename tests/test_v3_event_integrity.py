"""Stage 2-N checks stored v3 evidence without rewriting events or RSS."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tris_market_event_intelligence.collector import NewsStore, parse_rss
from tris_market_event_intelligence.event_store import EventProcessor
from tris_market_event_intelligence.v3_event_integrity import inspect_v3_event_integrity


def add_event(tmp_path: Path) -> tuple[Path, int]:
    dbpath = tmp_path / "news.sqlite3"
    store = NewsStore(dbpath)
    store.initialize()
    processor = EventProcessor(dbpath)
    activated, floor, created = processor.activate_v3_rollout()
    assert created and floor == 0
    now = datetime.now(UTC).isoformat()
    payload = (
        "<rss><channel><item><guid>meta-v3</guid>"
        "<title>Italy fines Meta €805m over accessing users' photos</title>"
        "<link>https://example.org/meta</link></item></channel></rss>"
    ).encode()
    assert store.save(
        "bbc_world", "https://example.org/feed", payload,
        parse_rss("bbc_world", payload, now), now, None, None,
    ) == 1
    result = processor.process_pending(
        extractor_version="multi-v3",
        received_since=activated, after_article_id=floor,
    )
    assert (result.processed, result.events) == (1, 1)
    with sqlite3.connect(dbpath) as conn:
        event_id = conn.execute(
            "SELECT id FROM economic_events"
        ).fetchone()[0]
    return dbpath, event_id


def test_v3_event_integrity_real_sqlite_links_and_readonly(tmp_path: Path) -> None:
    db, event_id = add_event(tmp_path)
    with sqlite3.connect(db) as conn:
        before = tuple(conn.iterdump())
    report = inspect_v3_event_integrity(db, as_of=datetime.now(UTC))
    assert report["status"] == "OBSERVED_STORED_EVIDENCE"
    assert report["post_cutover_v3_event_rows"] == 1
    assert report["consistent_event_rows"] == 1
    assert report["event_rows_with_integrity_issues"] == 0
    assert report["outside_fence_or_missing_article"] == 0
    assert report["event_status_counts"] == {"UNVERIFIED": 1}
    assert report["event_samples"][0]["event_id"] == event_id
    assert report["event_samples"][0]["integrity_issues"] == []
    assert report["receipt_to_event_seconds"]["count"] == 1
    assert report["source_authenticity_verified"] is False
    assert report["event_classification_accuracy_verified"] is False
    assert report["database_modified"] is False
    with sqlite3.connect(db) as conn:
        assert tuple(conn.iterdump()) == before


def test_v3_event_missing_checkpoint_and_tampered_hash_are_reported(
    tmp_path: Path,
) -> None:
    db, event_id = add_event(tmp_path)
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM extraction_runs WHERE extractor_version='multi-v3'")
        conn.execute(
            "UPDATE economic_events SET capture_sha256='incorrect' WHERE id=?",
            (event_id,),
        )
    report = inspect_v3_event_integrity(db, as_of=datetime.now(UTC))
    assert report["status"] == "INTEGRITY_ISSUES"
    assert report["post_cutover_v3_event_rows"] == 1
    assert report["consistent_event_rows"] == 0
    assert report["event_rows_with_integrity_issues"] == 1
    assert report["issue_counts"]["MISSING_V3_CHECKPOINT"] == 1
    assert report["issue_counts"]["CAPTURE_SHA256_MISMATCH"] == 1


def test_v3_event_missing_metadata_and_mismatched_source_are_reported(
    tmp_path: Path,
) -> None:
    db, event_id = add_event(tmp_path)
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM event_metadata WHERE event_id=?", (event_id,))
        conn.execute(
            "UPDATE economic_events SET source='unrelated' WHERE id=?",
            (event_id,),
        )
    report = inspect_v3_event_integrity(db, as_of=datetime.now(UTC))
    assert report["issue_counts"]["MISSING_EVENT_METADATA"] == 1
    assert report["issue_counts"]["ARTICLE_EVENT_PROVENANCE_MISMATCH"] == 1
    assert report["by_source"]["unrelated"]["integrity_issues"] == 1


def test_v3_audit_does_not_rewrite_future_or_historical_events(
    tmp_path: Path,
) -> None:
    db, event_id = add_event(tmp_path)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE economic_events SET processed_at=? WHERE id=?",
            ((datetime.now(UTC) + timedelta(days=1)).isoformat(), event_id),
        )
    report = inspect_v3_event_integrity(db, as_of=datetime.now(UTC))
    assert report["issue_counts"]["INVALID_PROCESSING_TIME"] == 1


def test_v3_audit_before_real_event_is_explicit_no_evidence(tmp_path: Path) -> None:
    db = tmp_path / "news.sqlite3"
    store = NewsStore(db)
    store.initialize()
    EventProcessor(db).activate_v3_rollout()
    report = inspect_v3_event_integrity(db, as_of=datetime.now(UTC))
    assert report["status"] == "NO_POST_CUTOVER_V3_EVENTS"
    assert report["post_cutover_v3_event_rows"] == 0
    assert report["event_classification_accuracy_verified"] is False


def test_v3_audit_missing_rollout_or_database_is_not_false_zero(
    tmp_path: Path,
) -> None:
    db = tmp_path / "news.sqlite3"
    with pytest.raises(ValueError, match="Existing"):
        inspect_v3_event_integrity(db, as_of=datetime.now(UTC))
    store = NewsStore(db)
    store.initialize()
    EventProcessor(db).initialize()
    assert inspect_v3_event_integrity(
        db, as_of=datetime.now(UTC),
    )["status"] == "NOT_ACTIVATED"
    with pytest.raises(ValueError, match="Timezone-aware"):
        inspect_v3_event_integrity(db, as_of=datetime(2026, 11, 15))
