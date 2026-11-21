"""Article discovery != article content acquisition (offline mock tests)."""
from pathlib import Path

import httpx

from tris_market_event_intelligence.collector import Article, NewsStore
from tris_market_event_intelligence.operations import run_cycle
from tris_market_event_intelligence.source_article_content import (
    allowed, capture, collect_full_text, extract_html,
)


CONTENT = (
    "The official article explains the economic event, its broader context, "
    "the latest developments and their implications for companies and markets. "
    "It compares demand, supply, prices and potential exposure to the news. "
    "The figures and statements are examples for an isolated mock test, "
    "not factual financial recommendations or genuine publisher reporting. "
) * 3


def seed(store: NewsStore, source: str, url: str, unique: str) -> None:
    store.initialize()
    article = Article(
        source=source, external_id=unique, url=url,
        title="Full economic news item", summary="Brief RSS metadata",
        published_at=None, detected_at="2026-11-21T00:00:00+00:00",
    )
    store.save(source, "https://example.com/feed", b"<rss/>",
               [article], article.detected_at, None, None)


def test_publisher_url_allowlist() -> None:
    assert allowed("https://www.bbc.com/news/articles/abc")
    assert allowed("https://www.federalreserve.gov/newsevents/pressreleases/x.htm")
    assert allowed("https://www.bok.or.kr/portal/bbs/B0000501/view.do?nttId=1")
    assert not allowed("http://www.bbc.com/news/articles/abc")
    assert not allowed("https://www.bbc.com.evil.example/news")
    assert not allowed("https://127.0.0.1/private")
    assert not allowed("https://www.bbc.com:8443/news")


def test_html_parser_preserves_article_and_discards_navigation() -> None:
    payload = (
        '<html><nav>Nothing useful from navigation</nav>'
        '<article><h1>Economic story</h1><p>' + CONTENT + '</p></article>'
        '<footer>Unrelated links</footer></html>'
    ).encode()
    body, attachments = extract_html(payload, "https://www.bbc.com/news/a")
    assert "Economic story" in body
    assert "companies and markets" in body
    assert "Nothing useful" not in body
    assert attachments == ()


def test_complete_body_is_durable_and_original_rss_untouched(tmp_path: Path) -> None:
    store = NewsStore(tmp_path / "news.sqlite3")
    url = "https://www.bbc.com/news/articles/same-item"
    seed(store, "bbc_world", url, "one")
    seed(store, "bbc_business", url, "two")
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(
            200, headers={"Content-Type": "text/html"},
            content=("<html><article>" + CONTENT + "</article></html>").encode(),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = collect_full_text(store.path, client=client, limit=4)
    assert result["complete"] == 2
    assert result["reused"] == 1
    assert len(requests) == 1
    with store.connect() as db:
        rows = db.execute("""
            SELECT a.summary, f.body, f.raw_payload, f.status, f.raw_sha256
            FROM articles a JOIN article_full_text f ON f.article_id=a.id
        """).fetchall()
    assert len(rows) == 2
    for summary, text, raw, status, digest in rows:
        assert summary == "Brief RSS metadata"
        assert "companies and markets" in text
        assert b"companies and markets" in raw
        assert status == "COMPLETE"
        assert len(digest) == 64


def test_incomplete_article_stays_pending_not_healthy(tmp_path: Path) -> None:
    store = NewsStore(tmp_path / "news.sqlite3")
    seed(store, "bbc_business", "https://www.bbc.com/news/articles/short", "short")

    with httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(
            200, text="<html><article>Only a title</article></html>",
            headers={"Content-Type": "text/html"},
        ),
    )) as client:
        result = collect_full_text(store.path, client=client, limit=2)
    assert result["status"] == "INCOMPLETE"
    assert result["incomplete"] == 1
    with store.connect() as db:
        row = db.execute(
            "SELECT status,error_code,retry_after FROM article_full_text"
        ).fetchone()
    assert row[0] == "INCOMPLETE"
    assert row[1] == "INSUFFICIENT_ARTICLE_TEXT"
    assert row[2]


def test_untrusted_url_never_contacts_network(tmp_path: Path) -> None:
    store = NewsStore(tmp_path / "news.sqlite3")
    seed(store, "bbc_world", "https://evil.example.invalid/private", "bad")
    with httpx.Client(transport=httpx.MockTransport(
        lambda _: (_ for _ in ()).throw(AssertionError("No network expected")),
    )) as client:
        result = collect_full_text(store.path, client=client)
    assert result["incomplete"] == 1


def test_operational_collection_degraded_if_article_body_missing(tmp_path: Path) -> None:
    feed = (
        '<rss><channel><item><title>Economic event</title>'
        '<link>https://www.bbc.com/news/articles/unavailable</link>'
        '<guid>test-event</guid><description>Summary only</description>'
        '</item></channel></rss>'
    ).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        if "feeds.bbci.co.uk" in str(request.url):
            return httpx.Response(200, content=feed)
        return httpx.Response(
            200, text="<html><article>Not enough text</article></html>",
            headers={"Content-Type": "text/html"},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = run_cycle(
            tmp_path / "news.sqlite3",
            selected_sources=("bbc_world",), client=client,
            enable_fulltext=True,
        )
    assert result["source_results"]["bbc_world"]["status"] == "ok"
    assert result["article_fulltext"]["incomplete"] == 1
    assert result["status"] == "degraded"
    assert result["new_articles"] == 1


def test_redirect_to_other_host_rejected() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "https://evil.test/"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        from tris_market_event_intelligence.source_article_content import ContentError

        try:
            capture(client, "https://www.bbc.com/news/a")
        except ContentError as exc:
            assert str(exc) == "UNTRUSTED_URL"
        else:
            raise AssertionError("Redirect should have been refused")
