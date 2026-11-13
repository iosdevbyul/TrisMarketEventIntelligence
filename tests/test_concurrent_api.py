"""Offline Phase 6-E: SQLite writer and read-only API share a live database safely."""

import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from tris_market_event_intelligence.api import create_app
from tris_market_event_intelligence.claim_store import ClaimProcessor
from tris_market_event_intelligence.collector import NewsStore, parse_rss

FEED = b"""<rss><channel>
<item><guid>item-1</guid><title>Fed rate announcement</title>
<link>https://example.org/news/1</link><description>Fed interest rate changed.</description></item>
<item><guid>item-2</guid><title>Inflation report</title>
<link>https://example.org/news/2</link><description>Consumer inflation accelerated.</description></item>
</channel></rss>"""


def _prepare(db: Path) -> int:
    store = NewsStore(db)
    store.initialize()
    articles = parse_rss("fed_all", FEED, "2026-11-14T00:00:00+00:00")
    store.save("fed_all", "https://example.org/feed", FEED, articles, "2026-11-14T00:00:00+00:00", None, None)
    ClaimProcessor(db).process_pending()
    with store.connect() as conn:
        return conn.execute("SELECT MAX(id) FROM article_event_claims").fetchone()[0]


def test_api_sees_only_committed_claims_and_keeps_cursor_stable(tmp_path: Path) -> None:
    db = tmp_path / "news.sqlite3"
    first_cursor = _prepare(db)
    writer = NewsStore(db).connect()
    try:
        with TestClient(create_app(db)) as client:
            initial = client.get("/api/v1/event-claims", params={"limit": 100})
            assert initial.status_code == 200
            initial_ids = [item["id"] for item in initial.json()["items"]]
            assert initial_ids
            assert initial.json()["nextCursor"] == first_cursor

            # Create a new claim in a write transaction without committing it.
            writer.execute("BEGIN IMMEDIATE")
            existing = writer.execute(
                "SELECT article_id, extractor_version, classification, tag, claim_kind, "
                "polarity, temporal_context, status, evidence, verification_note, source, "
                "source_url, published_at, detected_at, processed_at, capture_sha256 "
                "FROM article_event_claims ORDER BY id LIMIT 1"
            ).fetchone()
            assert existing is not None
            writer.execute(
                "INSERT INTO article_event_claims (article_id, extractor_version, ordinal, "
                "classification, tag, claim_kind, polarity, temporal_context, status, "
                "evidence, verification_note, source, source_url, published_at, detected_at, "
                "processed_at, capture_sha256) VALUES (?, ?, 99, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
                "?, ?, ?, ?, ?)",
                existing,
            )
            # An API read cannot see an uncommitted claim while the writer is active.
            during = client.get(
                "/api/v1/event-claims", params={"after_id": first_cursor}
            )
            assert during.status_code == 200
            assert during.json()["items"] == []

            writer.commit()
            after = client.get(
                "/api/v1/event-claims", params={"after_id": first_cursor}
            )
            assert after.status_code == 200
            assert len(after.json()["items"]) == 1
            assert after.json()["items"][0]["id"] > first_cursor
            assert client.get(
                "/api/v1/event-claims", params={"after_id": after.json()["nextCursor"]}
            ).json()["items"] == []
    finally:
        writer.rollback()
        writer.close()


def test_api_pagination_during_writer_transaction_never_duplicates(tmp_path: Path) -> None:
    db = tmp_path / "news.sqlite3"
    _prepare(db)
    with TestClient(create_app(db)) as client:
        cursor = 0
        found = []
        while True:
            response = client.get(
                "/api/v1/event-claims", params={"after_id": cursor, "limit": 1}
            )
            assert response.status_code == 200
            data = response.json()
            items = data["items"]
            found.extend(item["id"] for item in items)
            assert data["nextCursor"] >= cursor
            if not data["hasMore"]:
                break
            cursor = data["nextCursor"]
        assert found == sorted(set(found))
        assert found


def test_api_reads_dont_initialize_or_modify_database(tmp_path: Path) -> None:
    db = tmp_path / "absent.sqlite3"
    with TestClient(create_app(db)) as client:
        assert client.get("/ready/claims").status_code == 503
        assert client.get("/api/v1/event-claims").status_code == 503
    assert not db.exists()
    db.touch()
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE dummy (id INTEGER)")
    before = db.read_bytes()
    with TestClient(create_app(db)) as client:
        assert client.get("/ready/claims").status_code == 503
        assert client.get("/api/v1/event-claims").status_code == 503
    assert db.read_bytes() == before
