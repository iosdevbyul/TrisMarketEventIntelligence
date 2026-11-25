import sqlite3

import httpx
import pytest

from tris_market_event_intelligence.collector import Article, NewsStore
from tris_market_event_intelligence.source_article_content import collect_full_text


def test_official_scope_does_not_capture_media(tmp_path):
    path = tmp_path / "news.sqlite3"
    store = NewsStore(path)
    store.initialize()
    now = "2026-01-01T00:00:00+00:00"
    urls = {
        "bok_statistics": "https://www.bok.or.kr/report",
        "bbc_business": "https://www.bbc.co.uk/news/articles/example",
    }
    for source, url in urls.items():
        store.save(source, url, b"<rss/>", [
            Article(source, url, url, "Release", None, now, now)
        ], now, None, None)
    requested = []

    def handler(request):
        requested.append(str(request.url))
        return httpx.Response(503)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = collect_full_text(
            path, client=client, official_sources=("bok_statistics",),
        )
    assert requested == [urls["bok_statistics"]]
    assert result["attempted"] == 1
    assert result["missing_or_failed"] == 1
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM articles").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM article_full_text").fetchone()[0] == 1


def test_official_scope_rejects_unknown_sources(tmp_path):
    with pytest.raises(ValueError, match="Invalid official"):
        collect_full_text(tmp_path / "news.sqlite3", official_sources=("bbc_business",))
