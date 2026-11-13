"""Synthetic HTML mocks: no live portal requests are made in CI."""

from pathlib import Path

import httpx
import pytest

from tris_market_event_intelligence.collector import NewsStore
from tris_market_event_intelligence.web_scraper import (
    WebScrapingCollector,
    article_links,
    parse_article,
)


def _html(title: str = "NVIDIA announces new semiconductor factory supply") -> bytes:
    return f"""<html><head>
    <meta property="og:title" content="{title}">
    <meta property="article:published_time" content="2026-11-13T12:00:00+09:00">
    </head><body><div id="dic_area">The company announced its new factory.
    This development is presented as a fictional test case.
    It is only example text intended to exercise extraction.
    </div></body></html>""".encode()


def test_article_body_and_timestamp_extracted() -> None:
    article = parse_article(
        "naver", "https://news.naver.com/article/001/000001", _html(), "now",
    )
    assert article is not None
    assert "fictional test case" in (article.summary or "")
    assert article.published_at == "2026-11-13T03:00:00+00:00"


def test_links_are_portal_restricted_and_deduplicated() -> None:
    listing = b"""<a href="/article/001/000001?utm_source=a">Article A</a>
    <a href="/article/001/000001?utm_source=b">Article B</a>
    <a href="https://evil.example.org/article/001/000002">External</a>
    <a href="/section/101">Index</a>"""
    links = article_links("naver", "https://news.naver.com/section/101", listing)
    assert links == ["https://news.naver.com/article/001/000001"]


def test_robots_disallow_blocks_all_portal_html(tmp_path: Path) -> None:
    requested = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(
            200, text="User-agent: *\nDisallow: /",
            headers={"content-type": "text/plain"},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    collector = WebScrapingCollector(
        NewsStore(tmp_path / "db.sqlite3"), client, minimum_delay=1,
    )
    result = collector.collect_once(("naver",))
    assert result["naver"].status == "blocked_by_robots"
    assert requested == ["https://news.naver.com/robots.txt"]


def test_article_storage_dedup_and_body_evidence(tmp_path: Path) -> None:
    landing = b'<a href="/article/001/000001">A financial story</a>'
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /")
        if request.url.path == "/section/101":
            return httpx.Response(
                200, content=landing, headers={"content-type": "text/html"},
            )
        return httpx.Response(
            200, content=_html(), headers={"content-type": "text/html"},
        )

    store = NewsStore(tmp_path / "db.sqlite3")
    collector = WebScrapingCollector(
        store, httpx.Client(transport=httpx.MockTransport(handler)),
        max_articles=1, minimum_delay=1,
    )
    assert collector.collect_once(("naver",))["naver"].inserted == 1
    assert collector.collect_once(("naver",))["naver"].inserted == 0
    assert store.stats()["articles"] == 1
    with store.connect() as db:
        article = db.execute(
            "SELECT title, summary, source, capture_id FROM articles"
        ).fetchone()
        evidence = db.execute(
            "SELECT payload FROM captures WHERE id=?", (article[3],)
        ).fetchone()[0]
    assert article[2] == "scrape_naver"
    assert "fictional test" in article[1]
    assert b"fictional test" in evidence
    assert "https://news.naver.com/robots.txt" in calls


def test_fail_closed_when_robots_unavailable(tmp_path: Path) -> None:
    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(503)
    ))
    collector = WebScrapingCollector(NewsStore(tmp_path / "db.sqlite3"), client)
    assert collector.collect_once(("daum",))["daum"].status == "blocked_by_robots"


@pytest.mark.parametrize("max_articles,delay", [(0, 3), (31, 3), (1, 0)])
def test_bounded_crawl_configuration(tmp_path: Path, max_articles: int, delay: int) -> None:
    with pytest.raises(ValueError):
        WebScrapingCollector(
            NewsStore(tmp_path / "db.sqlite3"),
            max_articles=max_articles, minimum_delay=delay,
        )
