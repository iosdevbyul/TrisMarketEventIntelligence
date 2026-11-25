"""Provider registration and source-boundary regression."""
import httpx
import pytest

from tris_market_event_intelligence.collector import NewsStore, OfficialNewsCollector, parse_rss
from tris_market_event_intelligence.news_source_provider import (
    PUBLISHERS,
    accepts,
    canonical_url,
)
from tris_market_event_intelligence.source_article_content import capture, collect_full_text


def test_publisher_scope():
    assert accepts("cnbc_top", "https://www.cnbc.com/2026/11/24/story.html")
    assert not accepts("cnbc_top", "https://www.cnbc.com/video/clip")
    assert not accepts("cnbc_top", "https://www.hankyung.com/story")
    assert not accepts("hankyung_all", "https://www.hankyung.com:8443/story")
    assert canonical_url("https://www.bbc.com/news/a?utm_source=rss&id=1") == (
        "https://www.bbc.com/news/a?id=1"
    )


def test_provider_injection_and_text_only(tmp_path):
    class Fake:
        def fetch(self, client, url, headers):
            return httpx.Response(200, content=b"""<rss><channel>
              <item><title>Rate news</title><link>https://www.cnbc.com/story.html</link></item>
              <item><title>Video</title><link>https://www.cnbc.com/video/clip</link></item>
              <item><title>External</title><link>https://example.org/story</link></item>
            </channel></rss>""", request=httpx.Request("GET", url))
    store = NewsStore(tmp_path / "news.sqlite3")
    with httpx.Client() as client:
        result = OfficialNewsCollector(
            store, client, {"cnbc_top": PUBLISHERS["cnbc_top"].feed}, provider=Fake(),
        ).collect_once()
    assert result["cnbc_top"]["inserted"] == 1
    assert store.stats() == {"articles": 1, "captures": 1}


def test_new_publisher_original_capture():
    payload = b"<html><article><p>Company cuts profit guidance.</p></article></html>"
    with httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, content=payload, headers={"content-type": "text/html"})
    )) as client:
        row = capture(client, "https://www.hankyung.com/article/123", source="hankyung_all")
    assert row["body"] == "Company cuts profit guidance."
    assert row["raw_payload"] == payload


def test_registry_drives_backlog_and_coverage(tmp_path):
    from tris_market_event_intelligence.media_news_batch_cli import audit

    store = NewsStore(tmp_path / "news.sqlite3")
    store.initialize()
    for source in ("asiaeconomy_all", "sedaily_all", "edaily_all", "etnews_economy",
                   "zdnet_kr", "techcrunch_all", "marketwatch_top"):
        publisher = PUBLISHERS[source]
        url = f"https://{publisher.hosts[0]}/article/123"
        raw = f"<rss><channel><item><title>Profit news</title><link>{url}</link></item></channel></rss>".encode()
        observed = "2026-11-24T12:00:00+00:00"
        store.save(source, publisher.feed, raw, parse_rss(source, raw, observed),
                   observed, None, None)
    payload = b"<html><article><p>Company cuts profit guidance.</p></article></html>"
    with httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, content=payload, headers={"content-type": "text/html"})
    )) as client:
        result = collect_full_text(store.path, client=client, media_only=True)
    assert result["complete"] == 7
    assert result["pending_or_unverified"] == 0
    assert audit(store.path, "gemma4:e2b")["counts"] == {"READY_FOR_ANALYSIS": 7}



def test_edaily_cleartext_exception_is_exact():
    from tris_market_event_intelligence.news_source_provider import RSSNewsProvider
    from tris_market_event_intelligence.source_catalog import validate_feeds

    url = PUBLISHERS["edaily_all"].feed
    assert validate_feeds({"edaily_all": url}) == {"edaily_all": url}
    with pytest.raises(ValueError):
        validate_feeds({"other": url})
    with pytest.raises(ValueError):
        validate_feeds({"edaily_all": url + "?other=1"})
    assert not accepts("edaily_all", "http://www.edaily.co.kr/article/1")
    with httpx.Client(follow_redirects=True, transport=httpx.MockTransport(
        lambda _: httpx.Response(302, headers={"location": "https://example.org/"})
    )) as client:
        assert RSSNewsProvider().fetch(client, url, {}).status_code == 302


def test_zdnet_observed_article_region():
    payload = b"""<html><div class="news_head">Headline</div>
    <div id="articleBody"><div class="view_cont"><div id="content-123">
    <p>Company cuts profit guidance.</p></div></div>
    <div>Related stories</div></div><div class="right_cont">Other news</div></html>"""
    with httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(200, content=payload, headers={"content-type": "text/html"})
    )) as client:
        row = capture(client, "https://zdnet.co.kr/view/?no=123", source="zdnet_kr")
    assert row["body"] == "Company cuts profit guidance."
    assert row["raw_payload"] == payload


def test_access_restricted_not_retried_or_analyzed(tmp_path):
    from tris_market_event_intelligence.media_news_batch_cli import audit

    store = NewsStore(tmp_path / "news.sqlite3")
    store.initialize()
    source = "marketwatch_top"
    raw = b"<rss><channel><item><title>News</title><link>https://www.marketwatch.com/story/1</link></item></channel></rss>"
    observed = "2026-11-24T12:00:00+00:00"
    store.save(source, PUBLISHERS[source].feed, raw, parse_rss(source, raw, observed),
               observed, None, None)
    with httpx.Client(transport=httpx.MockTransport(
        lambda _: httpx.Response(401, content=b"Access restricted")
    )) as client:
        assert collect_full_text(store.path, client=client, media_only=True)["incomplete"] == 1
        report = audit(store.path, "gemma4:e2b")
        assert report["counts"] == {"ACCESS_RESTRICTED": 1}
        aid = report["pending"][0]["articleId"]
        assert collect_full_text(store.path, client=client, media_only=True,
                                 repair_ids=(aid,))["attempted"] == 0



def test_edaily_collector_accepts_legacy_xml(tmp_path):
    source = "edaily_all"
    raw = b"<rss><channel><item><title>Profit news</title><link>https://www.edaily.co.kr/news/read?newsId=123</link></item></channel></rss>"

    def transport(request):
        assert str(request.url) == PUBLISHERS[source].feed
        if request.headers["accept"] != "*/*":
            return httpx.Response(406)
        return httpx.Response(200, content=raw, headers={"content-type": "text/xml"})

    store = NewsStore(tmp_path / "news.sqlite3")
    with httpx.Client(headers={"Accept": "application/rss+xml, application/xml"},
                      transport=httpx.MockTransport(transport)) as client:
        result = OfficialNewsCollector(
            store, client, {source: PUBLISHERS[source].feed},
        ).collect_once()
    assert result[source] == {"status": "ok", "inserted": 1}
    assert store.stats() == {"articles": 1, "captures": 1}
