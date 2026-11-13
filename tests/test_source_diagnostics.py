"""Diagnostics for HTTP failures, portal policy preflight and extractor quality."""
from pathlib import Path

import httpx

from tris_market_event_intelligence.collector import NewsStore, OfficialNewsCollector
from tris_market_event_intelligence.discovery import SearchDiscovery
from tris_market_event_intelligence.live_probe import portal_robots_preflight


def test_http_403_and_429_diagnosed_without_urls(tmp_path: Path) -> None:
    responses = iter((httpx.Response(403), httpx.Response(429)))
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: next(responses)
    )) as client:
        store = NewsStore(tmp_path / "fail.db")
        collected = OfficialNewsCollector(
            store, client, {"bls_ppi": "https://www.bls.gov/feed/ppi.rss"}
        ).collect_once()
        searched = SearchDiscovery(
            store, client, gdelt_topics=("semiconductor",),
        ).collect_once(naver=False)
    assert collected["bls_ppi"]["status"] == "error"
    assert searched["gdelt_doc"].status == "rate_limited"
    with store.connect() as db:
        errors = dict(db.execute("SELECT source, last_error FROM fetch_state"))
    assert errors == {"bls_ppi": "HTTP_403", "gdelt_doc": "HTTP_429"}


def test_robots_preflight_never_fetches_article_html() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.host == "news.nate.com":
            return httpx.Response(200, text="User-agent: *\nDisallow: /")
        if request.url.host == "news.naver.com":
            return httpx.Response(200, text="User-agent: *\nAllow: /")
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = portal_robots_preflight(client)
    assert result["nate"]["landing_allowed"] == 0
    assert result["naver"]["landing_allowed"] == 1
    assert result["daum"]["status"] == "not_declared_permission_unknown"
    assert len(calls) == 3
    assert all(url.endswith("/robots.txt") for url in calls)


def test_rate_limited_source_stops_remaining_topics(tmp_path: Path) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(429)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = SearchDiscovery(
            NewsStore(tmp_path / "limited.db"), client,
            gdelt_topics=("stocks", "interest rates", "chips"),
        ).collect_once(naver=False)
    assert result["gdelt_doc"].status == "rate_limited"
    assert len(calls) == 1
