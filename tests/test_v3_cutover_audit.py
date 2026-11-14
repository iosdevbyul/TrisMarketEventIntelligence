"""Post-v3-cutover attribution is grounded in immutable stored evidence."""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tris_market_event_intelligence.collector import NewsStore, parse_rss
from tris_market_event_intelligence.event_store import EventProcessor
from tris_market_event_intelligence.v3_cutover_audit import audit_v3_cutover


def save_story(
    store: NewsStore, *, source: str, guid: str, title: str,
    url: str, received: str,
) -> None:
    payload = (
        f"<rss><channel><item><guid>{guid}</guid>"
        f"<title>{title}</title><link>{url.replace('&', '&amp;')}</link>"
        "</item></channel></rss>"
    ).encode()
    store.save(
        source, "https://example.org/rss", payload,
        parse_rss(source, payload, received), received, None, None,
    )


def test_cutover_audit_tracks_v3_v2_duplicate_pending_and_legacy(
    tmp_path: Path,
) -> None:
    dbpath = tmp_path / "news.sqlite3"
    store = NewsStore(dbpath)
    store.initialize()
    yesterday = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    save_story(
        store, source="bbc_world", guid="old", title="Old world news",
        url="https://example.org/old", received=yesterday,
    )
    processor = EventProcessor(dbpath)
    since, floor, created = processor.activate_v3_rollout()
    assert created and floor == 1
    now = datetime.now(UTC).isoformat()
    save_story(
        store, source="bbc_technology", guid="a",
        title="Italy fines Meta €805m over users' photos",
        url="https://example.org/meta?utm_source=tech", received=now,
    )
    save_story(
        store, source="bbc_world", guid="b",
        title="Italy fines Meta €805m over users' photos",
        url="https://example.org/meta?utm_source=world", received=now,
    )
    save_story(
        store, source="bbc_world", guid="c",
        title="A neighborhood festival opens today",
        url="https://example.org/community", received=now,
    )
    save_story(
        store, source="bbc_business", guid="d",
        title="Cloud service outage disrupts users",
        url="https://example.org/outage", received=now,
    )
    save_story(
        store, source="bbc_world", guid="e",
        title="Another unrelated local article",
        url="https://example.org/pending", received=now,
    )
    with sqlite3.connect(dbpath) as db:
        db.execute(
            "INSERT INTO extraction_runs VALUES (?,?,?)",
            (5, "multi-v2", now),
        )
    result = processor.process_pending(
        batch_size=3, extractor_version="multi-v3",
        after_article_id=floor, received_since=since,
    )
    assert (result.processed, result.events) == (3, 1)
    with sqlite3.connect(dbpath) as db:
        assert db.execute(
            "SELECT COUNT(*) FROM economic_events WHERE extractor_version='multi-v3'"
        ).fetchone()[0] == 1
    before = dbpath.read_bytes()
    report = audit_v3_cutover(dbpath, as_of=datetime.now(UTC))
    assert report["status"] == "OBSERVED_READ_ONLY"
    assert report["post_cutover_articles"] == 5
    assert report["stage_counts"] == {
        "V3_EVENT_WRITTEN": 1,
        "V3_DUPLICATE_CHECKPOINTED": 1,
        "V3_CLASSIFIED_NO_EVENT": 1,
        "V2_PROCESSED_FIRST": 1,
        "AWAITING_EXTRACTION": 1,
    }
    assert report["v3_event_rows"] == 1
    assert report["v2_event_rows"] == 0
    assert report["multiversion_event_anomalies"] == 0
    assert report["by_source"]["bbc_world"]["V3_DUPLICATE_CHECKPOINTED"] == 1
    samples = {row["article_id"]: row for row in report["article_samples"]}
    assert samples[3]["representative_article_id"] == 2
    assert samples[3]["stage"] == "V3_DUPLICATE_CHECKPOINTED"
    assert samples[5]["stage"] == "V2_PROCESSED_FIRST"
    assert samples[6]["stage"] == "AWAITING_EXTRACTION"
    assert report["cutover"]["article_id_floor"] == floor
    assert report["cutover"]["activated_at"] == since.isoformat()
    assert report["database_modified"] is False
    assert dbpath.read_bytes() == before


def test_before_activation_or_missing_table_never_claims_no_pending(
    tmp_path: Path,
) -> None:
    dbpath = tmp_path / "news.sqlite3"
    store = NewsStore(dbpath)
    store.initialize()
    EventProcessor(dbpath).initialize()
    status = audit_v3_cutover(dbpath, as_of=datetime.now(UTC))
    assert status == {
        "status": "NOT_ACTIVATED",
        "cutover": None,
        "database_modified": False,
    }
    with sqlite3.connect(dbpath) as db:
        db.execute(
            """CREATE TABLE event_extractor_rollout (
                extractor_version TEXT PRIMARY KEY,
                activated_at TEXT NOT NULL,
                article_id_floor INTEGER NOT NULL
            )"""
        )
        db.execute(
            "INSERT INTO event_extractor_rollout VALUES (?,?,?)",
            ("multi-v3", (datetime.now(UTC) + timedelta(days=1)).isoformat(), 0),
        )
    status = audit_v3_cutover(dbpath, as_of=datetime.now(UTC))
    assert status["status"] == "INVALID_CUTOVER"


def test_invalid_input_does_not_create_database(tmp_path: Path) -> None:
    path = tmp_path / "missing.sqlite3"
    with pytest.raises(ValueError, match="Existing regular"):
        audit_v3_cutover(path, as_of=datetime.now(UTC))
    assert not path.exists()
    path.touch()
    with pytest.raises(ValueError, match="timezone-aware"):
        audit_v3_cutover(path, as_of=datetime(2026, 11, 15))
