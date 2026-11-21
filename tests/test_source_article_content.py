"""Article discovery != article content acquisition (offline mock tests)."""
from pathlib import Path

import httpx

from tris_market_event_intelligence.collector import Article, NewsStore
from tris_market_event_intelligence.operations import run_cycle
from tris_market_event_intelligence.source_article_content import (
    allowed,
    capture,
    collect_full_text,
    extract_html,
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


def test_nested_main_and_article_blocks_are_not_prematurely_closed() -> None:
    payload = (
        "<html><main><div><div><span>Intro</span></div>"
        "<section><p>" + CONTENT + "</p><p>Conclusion.</p></section>"
        "</div></main></html>"
    ).encode()
    body, _ = extract_html(payload, "https://www.federalreserve.gov/a")
    assert "companies and markets" in body
    assert "Conclusion." in body


def test_priority_prefers_article_over_website_main_navigation() -> None:
    payload = (
        "<html><main><div>Irrelevant content " + CONTENT + "</div>"
        "<article><p>Only the report: " + CONTENT + "</p></article>"
        "</main></html>"
    ).encode()
    body, _ = extract_html(payload, "https://www.bbc.com/news/article/1")
    assert body.startswith("Only the report:")
    assert "Irrelevant content" not in body


def test_release_page_prioritizes_longer_linked_pdf(monkeypatch: object) -> None:
    from tris_market_event_intelligence import source_article_content as fulltext

    html = (
        "<html><article>" + CONTENT +
        '<a href="/files/release.pdf">Full release PDF</a></article></html>'
    ).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(".pdf"):
            return httpx.Response(200, content=b"%PDF-1.4 fixture")
        return httpx.Response(
            200, content=html, headers={"Content-Type": "text/html"},
        )

    monkeypatch.setattr(
        fulltext, "extract_pdf",
        lambda payload: ("The complete release. " * 160, ()),
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = fulltext.capture(
            client, "https://www.federalreserve.gov/press/release"
        )
    assert result["status"] == "COMPLETE"
    assert result["format"] == "PDF"
    assert result["source_url"].endswith(".pdf")
    assert len(result["body"]) > len(CONTENT)


def test_bbc_video_page_is_not_a_news_article() -> None:
    from tris_market_event_intelligence.source_article_content import ContentError

    with httpx.Client(transport=httpx.MockTransport(
        lambda _: (_ for _ in ()).throw(AssertionError("Must not fetch video")),
    )) as client:
        try:
            capture(client, "https://www.bbc.co.uk/iplayer/episode/m0034tjc")
        except ContentError as exc:
            assert str(exc) == "NON_ARTICLE_MEDIA_PAGE"
        else:
            raise AssertionError("iPlayer page should not pass as article")


def test_non_article_media_is_terminal_not_a_retriable_failure(
    tmp_path: Path,
) -> None:
    store = NewsStore(tmp_path / "news.sqlite3")
    seed(
        store, "bbc_technology",
        "https://www.bbc.co.uk/iplayer/episode/m0034tjc",
        "non-article",
    )
    with httpx.Client(transport=httpx.MockTransport(
        lambda _: (_ for _ in ()).throw(AssertionError("Must not fetch video")),
    )) as client:
        first = collect_full_text(store.path, client=client, limit=2)
        second = collect_full_text(store.path, client=client, limit=2)
    assert first["non_article"] == 1
    assert first["incomplete"] == 0
    assert first["pending_or_unverified"] == 0
    assert second["attempted"] == 0
    with store.connect() as db:
        assert db.execute(
            "SELECT status,error_code,retry_after FROM article_full_text"
        ).fetchone() == ("NON_ARTICLE", "NON_ARTICLE_MEDIA_PAGE", None)


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
        b'<rss><channel><item><title>Economic event</title>'
        b'<link>https://www.bbc.com/news/articles/unavailable</link>'
        b'<guid>test-event</guid><description>Summary only</description>'
        b'</item></channel></rss>'
    )

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

def test_market_impact_model_receives_verified_body_not_rss_only(
    tmp_path: Path,
) -> None:
    from tris_market_event_intelligence.event_store import EventProcessor
    from tris_market_event_intelligence.impact import ImpactResult
    from tris_market_event_intelligence.impact_store import ImpactProcessor
    from tris_market_event_intelligence.source_article_content import initialize

    store = NewsStore(tmp_path / "events.sqlite3")
    seed(
        store, "bbc_business", "https://www.bbc.com/news/articles/market-impact",
        "impact-verified",
    )
    initialize(store)
    processor = EventProcessor(store.path)
    processor.initialize()
    with store.connect() as db:
        db.execute("""
            INSERT INTO economic_events (
                article_id, extractor_version, classification, status,
                direction, change_basis_points, evidence, source,
                source_url, published_at, detected_at, processed_at,
                capture_sha256
            ) VALUES (
                1, 'test', 'CORPORATE_EVENT', 'CANDIDATE', NULL, NULL,
                'Brief RSS metadata', 'bbc_business',
                'https://www.bbc.com/news/articles/market-impact', NULL,
                '2026-11-21T00:00:00+00:00', '2026-11-21T00:00:00+00:00',
                'sha'
            )
        """)
        db.execute("""
            INSERT INTO event_metadata (
                event_id, claim_kind, tags_json, verification_note
            ) VALUES (1, 'REPORTED', '[]', 'test')
        """)
        import hashlib

        db.execute("""
            INSERT INTO article_full_text (
                article_id, status, observed_at, content_url, body_sha256,
                raw_sha256, raw_payload, body, format, sparse_pages
            ) VALUES (1, 'COMPLETE', '2026-11-21T00:00:00+00:00',
                      'https://www.bbc.com/news/articles/market-impact',
                      ?, 'rawsha', X'4142', ?, 'HTML', '[]')
        """, (hashlib.sha256(CONTENT.encode()).hexdigest(), CONTENT))

    class Recorder:
        model_id = "verified-body-fixture"

        def __init__(self) -> None:
            self.evidence = ""

        def analyze(self, event: object) -> ImpactResult:
            self.evidence = event.evidence
            return ImpactResult(impacts=[])

    provider = Recorder()
    counts = ImpactProcessor(store.path, provider).process_pending()
    assert counts.processed == 1
    assert "[ORIGINAL_SOURCE_DOCUMENT]" in provider.evidence
    assert "companies and markets" in provider.evidence
    assert "Brief RSS metadata" in provider.evidence


def test_unreviewed_pdf_document_is_not_sent_to_model(tmp_path: Path) -> None:
    from tris_market_event_intelligence.event_store import EventProcessor
    from tris_market_event_intelligence.impact import ImpactResult
    from tris_market_event_intelligence.impact_store import ImpactProcessor
    from tris_market_event_intelligence.source_article_content import initialize

    store = NewsStore(tmp_path / "pdf.sqlite3")
    seed(store, "bok_statistics", "https://www.bok.or.kr/portal/article", "pdf")
    initialize(store)
    EventProcessor(store.path).initialize()
    with store.connect() as db:
        db.execute("""
            INSERT INTO economic_events (
                article_id, extractor_version, classification, status,
                evidence, source, source_url, detected_at,
                processed_at, capture_sha256
            ) VALUES (1,'test','ECONOMIC_INDICATOR','CANDIDATE',
                      'Economic release','bok_statistics',
                      'https://www.bok.or.kr/portal/article',
                      '2026-11-21T00:00:00+00:00',
                      '2026-11-21T00:00:00+00:00', 'sha')
        """)
        db.execute("""
            INSERT INTO event_metadata (event_id,claim_kind,tags_json,verification_note)
            VALUES (1,'REPORTED','[]','test')
        """)
        db.execute("""
            INSERT INTO article_full_text (
                article_id,status,observed_at,body,body_sha256,sparse_pages
            ) VALUES (1,'REVIEW_REQUIRED','2026-11-21T00:00:00+00:00',
                      ?, 'sha','[2]')
        """, (CONTENT,))

    class ForbiddenProvider:
        model_id = "pdf-review-fixture"

        def analyze(self, event: object) -> ImpactResult:
            raise AssertionError("Unreviewed numbers should not reach model")

    result = ImpactProcessor(store.path, ForbiddenProvider()).process_pending()
    assert result.deferred == 1
    assert result.processed == 0
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM impact_runs").fetchone()[0] == 0


def test_continuous_backlog_catches_old_and_new_articles(
    tmp_path: Path,
) -> None:
    store = NewsStore(tmp_path / "continuous.sqlite3")
    for index in range(18):
        seed(
            store, "bbc_world",
            f"https://www.bbc.com/news/articles/continuous-{index}",
            f"continuous-{index}",
        )
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html"},
            content=("<html><article>" + CONTENT + "</article></html>").encode(),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        first = collect_full_text(store.path, client=client, limit=12)
        second = collect_full_text(store.path, client=client, limit=12)
        third = collect_full_text(store.path, client=client, limit=12)
    assert first["attempted"] == 12
    assert first["missing_or_failed"] == 6
    assert first["status"] == "INCOMPLETE"
    assert second["attempted"] == 6
    assert second["missing_or_failed"] == 0
    assert second["status"] == "COMPLETE"
    assert third["attempted"] == 0
    assert len(seen) == 18
    assert len(set(seen)) == 18
    assert any("continuous-0" in url for url in seen[:12])
    assert any("continuous-17" in url for url in seen[:12])


def test_manual_collector_reuses_normalized_fulltext_queue(
    tmp_path: Path, monkeypatch: object, capsys: object,
) -> None:
    import json

    from tris_market_event_intelligence import collector, source_article_content

    recorded: list[Path] = []
    monkeypatch.setattr(
        collector.OfficialNewsCollector,
        "collect_once",
        lambda self: {"bbc_world": {"status": "ok", "inserted": 0}},
    )
    monkeypatch.setattr(
        source_article_content, "collect_full_text",
        lambda path: recorded.append(path) or {"status": "COMPLETE", "attempted": 0},
    )
    database = tmp_path / "manual.sqlite3"
    collector.run_collector(
        database, 300, True,
        feeds={"bbc_world": "https://feeds.bbci.co.uk/news/world/rss.xml"},
        enable_fulltext=True,
    )
    assert recorded == [database]
    report = json.loads(capsys.readouterr().out)
    assert report["rss"]["bbc_world"]["status"] == "ok"
    assert report["fulltext"]["status"] == "COMPLETE"
