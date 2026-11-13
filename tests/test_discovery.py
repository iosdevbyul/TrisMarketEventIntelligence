"""Offline mocked discovery and article-grouping integration tests."""

import json
from pathlib import Path

import httpx
import pytest

from tris_market_event_intelligence.collector import NewsStore
from tris_market_event_intelligence.discovery import (
    SearchDiscovery,
    parse_gdelt,
    parse_naver,
)
from tris_market_event_intelligence.story_index import (
    StoryIndex,
    canonical_url,
    title_fingerprint,
)


def test_search_adapters_normalize_dates_and_strip_markup() -> None:
    naver = {
        "items": [{"title": "<b>반도체</b> 투자 확대", "description": "<b>AI</b> 투자",
                   "originallink": "https://publisher.example.org/one?utm_source=n",
                   "link": "https://news.naver.com/a",
                   "pubDate": "Fri, 13 Nov 2026 18:00:00 +0900"}]
    }
    results = parse_naver(json.dumps(naver).encode(), "2026-11-14T00:00:00+00:00")
    assert results[0].title == "반도체 투자 확대"
    assert results[0].published_at == "2026-11-13T09:00:00+00:00"
    assert results[0].url.startswith("https://publisher.example.org/")
    gdelt = {"articles": [{
        "title": "Semiconductor export controls expanded",
        "url": "https://wire.example.com/entry", "seendate": "20261113T180000Z",
    }]}
    items = parse_gdelt(json.dumps(gdelt).encode(), "now")
    assert items[0].published_at == "2026-11-13T18:00:00+00:00"


def test_optional_naver_and_source_error_isolation(tmp_path: Path) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.host == "openapi.naver.com":
            assert request.headers["x-naver-client-id"] == "test-id"
            assert request.headers["x-naver-client-secret"] == "test-secret"
            return httpx.Response(200, json={"items": [{
                "title": "Semiconductor factory supply chain",
                "originallink": "https://wire.example.org/1",
                "pubDate": "Fri, 13 Nov 2026 18:00:00 GMT",
            }]})
        return httpx.Response(429)

    store = NewsStore(tmp_path / "news.db")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    discovery = SearchDiscovery(
        store, client, "test-id", "test-secret",
        naver_topics=("semiconductors",), gdelt_topics=("geopolitics",),
    )
    result = discovery.collect_once()
    assert result["naver_news"].inserted == 1
    assert result["gdelt_doc"].status == "error"
    assert discovery.collect_once()["naver_news"].inserted == 0
    assert store.stats()["articles"] == 1
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM captures").fetchone()[0] == 1
        assert db.execute(
            "SELECT last_error FROM fetch_state WHERE source='gdelt_doc'"
        ).fetchone()[0] == "HTTPStatusError"
    assert len(calls) == 4


def test_missing_naver_credentials_skip_without_network(tmp_path: Path) -> None:
    store = NewsStore(tmp_path / "db.sqlite3")
    discovery = SearchDiscovery(store, client=httpx.Client(
        transport=httpx.MockTransport(lambda request: pytest.fail(str(request)))
    ))
    assert discovery.collect_once(naver=True, gdelt=False)["naver_news"].status == (
        "not_configured"
    )


def test_strict_url_and_title_dedup(tmp_path: Path) -> None:
    path = tmp_path / "news.db"
    store = NewsStore(path)
    store.initialize()
    raw = b"source-capture"
    from tris_market_event_intelligence.collector import Article

    articles = [
        Article("bbc_business", "1", "https://site.example.com/news/a?utm_source=x",
                "Major company announces global factory expansion plan", None,
                "2026-11-13T00:00:00+00:00", "now"),
        Article("naver_news", "2", "https://site.example.com/news/a?utm_medium=y",
                "Major company announces global factory expansion plan", None,
                "2026-11-13T00:00:00+00:00", "now"),
        Article("gdelt_doc", "3", "https://another.example.com/news/3",
                "Major company announces global factory expansion plan", None,
                "2026-11-13T05:00:00+00:00", "now"),
        Article("gdelt_doc", "4", "https://another.example.com/news/4",
                "Major company announces global factory expansion plan", None,
                "2026-11-14T05:00:00+00:00", "now"),
        Article("bbc_world", "5", "https://another.example.com/news/5",
                "Short headline", None, None, "now"),
    ]
    for item in articles:
        store.save(item.source, "https://example.org/feed", raw, [item], "now", None, None)
    index = StoryIndex(path)
    stats = index.index_pending()
    assert stats.indexed == 5
    assert stats.new_groups == 3
    assert stats.url_matches == 1
    assert stats.title_matches == 1
    assert index.index_pending().indexed == 0
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM articles").fetchone()[0] == 5
        groups = db.execute(
            "SELECT story_id FROM story_members ORDER BY article_id"
        ).fetchall()
        assert [group[0] for group in groups][:3] == [groups[0][0]] * 3


def test_canonical_url_preserves_meaningful_query_and_dates() -> None:
    assert canonical_url("https://EXAMPLE.com/news/?b=2&utm_campaign=x&a=1#part") == (
        "https://example.com/news?a=1&b=2"
    )
    assert title_fingerprint("Short", "2026-11-13T00:00:00+00:00") is None
    assert title_fingerprint("A long enough financial news article headline", None) is None
