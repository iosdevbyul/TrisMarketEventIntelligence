"""Discovery first-seen timeline and Naver/GDELT telemetry regressions."""
from datetime import UTC, datetime
from pathlib import Path

import httpx

from tris_market_event_intelligence.collector import NewsStore
from tris_market_event_intelligence.discovery import SearchDiscovery
from tris_market_event_intelligence.news_latency import audit_news_latency


def test_both_providers_use_response_receipt_not_request_start(tmp_path: Path) -> None:
    call_instants: list[datetime] = []

    def handler(request: httpx.Request) -> httpx.Response:
        call_instants.append(datetime.now(UTC))
        if request.url.host == "openapi.naver.com":
            return httpx.Response(200, json={"items": [{
                "title": "Company earnings announcement",
                "originallink": "https://publisher.example/news/1",
                "pubDate": "Sat, 14 Nov 2026 01:00:00 GMT",
            }]})
        return httpx.Response(200, json={"articles": [{
            "title": "Semiconductor export change",
            "url": "https://publisher.example/news/2",
            "seendate": "20261114T010000Z",
        }]})

    store = NewsStore(tmp_path / "news.db")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = SearchDiscovery(
            store, client, "id", "secret",
            naver_topics=("stocks",), gdelt_topics=("technology",),
        ).collect_once()
    assert result["naver_news"].inserted == 1
    assert result["gdelt_doc"].inserted == 1
    with store.connect() as db:
        first_seen = db.execute(
            "SELECT source,detected_at FROM articles ORDER BY source"
        ).fetchall()
        attempts = db.execute(
            "SELECT source,request_started_at,response_at,completed_at,"
            "request_elapsed_ms,status,returned,inserted FROM fetch_attempts "
            "ORDER BY source"
        ).fetchall()
    assert len(first_seen) == len(attempts) == 2
    for (source, detected), row in zip(first_seen, attempts, strict=True):
        assert source == row[0]
        assert detected == row[2]
        assert datetime.fromisoformat(row[1]) <= datetime.fromisoformat(row[2])
        assert datetime.fromisoformat(row[2]) <= datetime.fromisoformat(row[3])
        assert datetime.fromisoformat(row[2]) >= min(call_instants)
        assert row[4] >= 0
        assert row[5:] == ("ok", 1, 1)


def test_duplicate_discovery_preserves_original_first_seen(
    tmp_path: Path,
) -> None:
    store = NewsStore(tmp_path / "news.db")
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"items": [{
            "title": "Important market notice",
            "originallink": "https://publisher.example/once",
        }]})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        collector = SearchDiscovery(
            store, client, "id", "secret",
            naver_topics=("a",), gdelt_topics=(),
        )
        assert collector.collect_once(gdelt=False)["naver_news"].inserted == 1
        with store.connect() as db:
            original = db.execute("SELECT detected_at FROM articles").fetchone()[0]
        assert collector.collect_once(gdelt=False)["naver_news"].inserted == 0
    with store.connect() as db:
        after = db.execute("SELECT detected_at FROM articles").fetchone()[0]
        attempts = db.execute(
            "SELECT returned,inserted,status FROM fetch_attempts ORDER BY id"
        ).fetchall()
    assert original == after
    assert attempts == [(1, 1, "ok"), (1, 0, "ok")]


def test_rate_limit_stops_remaining_topics_and_records_failed_attempt(
    tmp_path: Path,
) -> None:
    calls = []

    def blocked(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(429)
    store = NewsStore(tmp_path / "news.db")
    with httpx.Client(transport=httpx.MockTransport(blocked)) as client:
        result = SearchDiscovery(
            store, client, "id", "secret",
            naver_topics=("one", "two"), gdelt_topics=(),
        ).collect_once(gdelt=False)
    assert result["naver_news"].status == "rate_limited"
    assert len(calls) == 1
    with store.connect() as db:
        row = db.execute(
            "SELECT response_at,status,returned,inserted "
            "FROM fetch_attempts"
        ).fetchone()
        error = db.execute(
            "SELECT last_error FROM fetch_state WHERE source='naver_news'"
        ).fetchone()[0]
    assert datetime.fromisoformat(row[0]).tzinfo is not None
    assert row[1:] == ("error", 0, 0)
    assert error == "HTTP_429"


def test_missing_naver_credentials_never_emits_fake_request_telemetry(
    tmp_path: Path,
) -> None:
    store = NewsStore(tmp_path / "news.db")
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(500),
    )) as client:
        result = SearchDiscovery(store, client).collect_once(naver=True, gdelt=False)
    assert result["naver_news"].status == "not_configured"
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM fetch_attempts").fetchone()[0] == 0


def test_existing_latency_audit_includes_discovery_source_stats(tmp_path: Path) -> None:
    store = NewsStore(tmp_path / "news.db")
    with httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, json={"articles": [{
            "title": "Economic research source",
            "url": "https://publisher.example/1",
            "seendate": "20261114T010000Z",
        }]}),
    )) as client:
        SearchDiscovery(
            store, client, gdelt_topics=("economics",),
        ).collect_once(naver=False, gdelt=True)
    report = audit_news_latency(store.path, as_of=datetime.now(UTC))
    source = {entry["source"]: entry for entry in report["sources"]}["gdelt_doc"]
    assert source["articles"] == 1
    assert source["fetch_attempts"] == 1
    assert source["fetch_request_duration_count"] == 1
    assert source["publish_to_first_seen_valid_count"] == 1
    assert report["real_time_detection_verified"] is False
