"""Provider registration and source-boundary regression."""
import httpx

from tris_market_event_intelligence.collector import NewsStore, OfficialNewsCollector
from tris_market_event_intelligence.news_source_provider import (
    PUBLISHERS,
    accepts,
    canonical_url,
)
from tris_market_event_intelligence.source_article_content import capture


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
