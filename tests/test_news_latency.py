"""Synthetic RSS/SQLite news arrival-time and monitoring regressions."""
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from tris_market_event_intelligence.collector import NewsStore, OfficialNewsCollector
from tris_market_event_intelligence.news_latency import audit_news_latency

RSS = (
    b'<rss version="2.0"><channel><item>'
    b'<title>Market notice</title><guid>one</guid>'
    b'<link>https://example.org/one</link>'
    b'<pubDate>Mon, 16 Nov 2026 10:00:00 GMT</pubDate>'
    b'</item></channel></rss>'
)


def test_first_seen_after_response_and_durable_timing(tmp_path: Path) -> None:
    observed = []

    def response(_: httpx.Request) -> httpx.Response:
        observed.append(datetime.now(UTC))
        return httpx.Response(200, content=RSS)

    store = NewsStore(tmp_path / "rss.sqlite3")
    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        collector = OfficialNewsCollector(store, client, {"fed": "https://example.org/rss"})
        assert collector.collect_once()["fed"] == {"status": "ok", "inserted": 1}
    with store.connect() as db:
        first_seen = db.execute("SELECT detected_at FROM articles").fetchone()[0]
        started, response_at, elapsed, status = db.execute(
            "SELECT request_started_at,response_at,request_elapsed_ms,status "
            "FROM fetch_attempts",
        ).fetchone()
        fetched = db.execute("SELECT fetched_at FROM captures").fetchone()[0]
    assert datetime.fromisoformat(started) <= observed[0]
    assert datetime.fromisoformat(response_at) >= observed[0]
    assert datetime.fromisoformat(first_seen) >= observed[0]
    assert first_seen == fetched == response_at
    assert elapsed >= 0
    assert status == "ok"


def test_repeat_retains_original_first_seen_and_tracks_304(tmp_path: Path) -> None:
    store = NewsStore(tmp_path / "rss.sqlite3")
    responses = iter([
        httpx.Response(200, content=RSS, headers={"ETag": '"a"'}),
        httpx.Response(304),
    ])
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: next(responses)),
    ) as client:
        collector = OfficialNewsCollector(store, client, {"fed": "https://example.org/rss"})
        collector.collect_once()
        with store.connect() as db:
            prior = db.execute("SELECT detected_at FROM articles").fetchone()[0]
        collector.collect_once()
    with store.connect() as db:
        after = db.execute("SELECT detected_at FROM articles").fetchone()[0]
        statuses = db.execute("SELECT status FROM fetch_attempts ORDER BY id").fetchall()
    assert after == prior
    assert statuses == [("ok",), ("not_modified",)]


def test_errors_and_recovery_are_counted(tmp_path: Path) -> None:
    store = NewsStore(tmp_path / "rss.sqlite3")
    responses = iter([httpx.Response(503), httpx.Response(200, content=RSS)])
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: next(responses)),
    ) as client:
        collector = OfficialNewsCollector(store, client, {"fed": "https://example.org/rss"})
        assert collector.collect_once()["fed"]["status"] == "error"
        assert collector.collect_once()["fed"]["status"] == "ok"
    with store.connect() as db:
        rows = db.execute(
            "SELECT status,returned,inserted FROM fetch_attempts ORDER BY id"
        ).fetchall()
    assert rows == [("error", 0, 0), ("ok", 1, 1)]


def test_audit_separates_missing_and_future_publication(tmp_path: Path) -> None:
    store = NewsStore(tmp_path / "rss.sqlite3")
    store.initialize()
    as_of = datetime(2026, 11, 18, tzinfo=UTC)
    with store.connect() as db:
        db.execute(
            "INSERT INTO captures(source,feed_url,fetched_at,sha256,payload) "
            "VALUES ('fed','https://example.org/rss',?,'fake',X'00')",
            ((as_of - timedelta(hours=1)).isoformat(),),
        )
        for identifier, published, seen in [
            ("a", "2026-11-16T10:00:00+00:00", "2026-11-16T10:00:30+00:00"),
            ("b", None, "2026-11-16T10:01:00+00:00"),
            ("c", "2026-11-16T10:03:00+00:00", "2026-11-16T10:02:00+00:00"),
        ]:
            db.execute(
                "INSERT INTO articles(source,external_id,url,title,published_at,"
                "detected_at,capture_id) VALUES ('fed',?,'https://example.org/a',"
                "'Example',?,?,1)",
                (identifier, published, seen),
            )
    report = audit_news_latency(store.path, as_of=as_of)
    group = report["sources"][0]
    assert group["articles"] == 3
    assert group["published_timestamp_missing"] == 1
    assert group["published_timestamp_future_of_first_seen"] == 1
    assert group["publish_to_first_seen_valid_count"] == 1
    assert group["publish_to_first_seen_median_seconds"] == 30
    assert report["real_time_detection_verified"] is False


def test_legacy_db_audit_does_not_create_missing_tables(tmp_path: Path) -> None:
    store = NewsStore(tmp_path / "rss.sqlite3")
    store.initialize()
    with store.connect() as db:
        db.execute("DROP TABLE fetch_attempts")
    result = audit_news_latency(store.path, as_of=datetime.now(UTC))
    assert result["fetch_attempt_telemetry_available"] is False
    with store.connect() as db:
        assert db.execute(
            "SELECT count(*) FROM sqlite_master WHERE name='fetch_attempts'",
        ).fetchone()[0] == 0


def test_audit_refuses_missing_db_and_naive_clock(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Existing"):
        audit_news_latency(tmp_path / "missing.sqlite3", as_of=datetime.now(UTC))
    store = NewsStore(tmp_path / "existing.sqlite3")
    store.initialize()
    with pytest.raises(ValueError, match="timezone"):
        audit_news_latency(store.path, as_of=datetime(2026, 11, 16))
