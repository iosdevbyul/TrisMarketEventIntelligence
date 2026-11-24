import base64
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from tris_market_event_intelligence.collector import Article, NewsStore
from tris_market_event_intelligence.sec_operational_content import (
    canonical_index,
    documents,
    index_identity,
)
from tris_market_event_intelligence.source_article_content import capture, capture_sec, collect_full_text

ROOT = "https://www.sec.gov/Archives/edgar/data/789019/000078901926000001/"
INDEX = ROOT + "0000789019-26-000001-index.htm"
PRIMARY = ROOT + "report.htm"
EXHIBIT = ROOT + "release.htm"
TABLE = b"""<html><table>
<tr><td>1</td><td>Filing</td><td><a href="report.htm">report.htm</a></td><td>8-K</td></tr>
<tr><td>2</td><td>Release</td><td><a href="release.htm">release.htm</a></td><td>EX-99.1</td></tr>
</table></html>"""
BODY = b"<html><body>Issuer reported its quarterly results.</body></html>"
RELEASE = b"<html><body>Revenue increased 10 percent.</body></html>"


def test_issuer_neutral_document_resolution():
    assert index_identity(INDEX) == ("789019", "0000789019-26-000001")
    assert documents(TABLE, INDEX) == (PRIMARY, (EXHIBIT,))
    assert index_identity(INDEX.replace("000078901926000001/", "000078901926000002/")) is None
    with pytest.raises(ValueError, match="AMBIGUOUS"):
        documents(TABLE.replace(b'href="report.htm"', b'href="https://other.test/report.htm"'), INDEX)


def test_capture_preserves_originals_and_exhibits(monkeypatch):
    monkeypatch.setattr("tris_market_event_intelligence.source_article_content.time.sleep", lambda _: None)
    payloads = {INDEX: TABLE, PRIMARY: BODY, EXHIBIT: RELEASE}
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=payloads[str(request.url)])
    )) as client:
        row = capture(client, INDEX)
    assert row["format"] == "SEC_BUNDLE"
    assert row["status"] == "COMPLETE"
    assert "Revenue increased 10 percent." in row["body"]
    bundle = json.loads(row["raw_payload"])
    assert base64.b64decode(bundle["indexRawBase64"]) == TABLE
    assert hashlib.sha256(row["raw_payload"]).hexdigest() == row["raw_sha256"]
    for doc in bundle["documents"]:
        assert base64.b64decode(doc["rawBase64"]) == payloads[doc["url"]]


def test_sec_queue_isolated_and_idempotent(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("tris_market_event_intelligence.source_article_content.time.sleep", lambda _: None)
    store = NewsStore(tmp_path / "news.sqlite3")
    store.initialize()
    for source, url in [("sec_8k_789019", INDEX), ("bbc_world", "https://www.bbc.com/news/a")]:
        article = Article(source=source, external_id=source, url=url, title="Disclosure",
                          summary=None, published_at=None,
                          detected_at="2026-11-21T00:00:00+00:00")
        store.save(source, "https://example.com/feed", b"<rss/>", [article],
                   article.detected_at, None, None)
    payloads = {INDEX: TABLE, PRIMARY: BODY, EXHIBIT: RELEASE}
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=payloads[str(request.url)])
    )) as client:
        first = collect_full_text(store.path, client=client, sec_only=True)
        second = collect_full_text(store.path, client=client, sec_only=True)
    assert first["complete"] == 1
    assert first["missing_or_failed"] == 0
    assert second["attempted"] == 0
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM article_full_text").fetchone()[0] == 1


@pytest.mark.parametrize("url", [PRIMARY, INDEX, "https://www.sec.gov/ix?doc=" + PRIMARY.removeprefix("https://www.sec.gov")])
def test_canonical_index_for_operational_feed_links(url):
    assert canonical_index(url, "sec_8k_789019") == INDEX
    with pytest.raises(ValueError, match="CIK_MISMATCH"):
        canonical_index(url, "sec_8k_1045810")


def test_retry_previous_generic_failure_with_direct_document(tmp_path, monkeypatch):
    from tris_market_event_intelligence.source_article_content import initialize

    monkeypatch.setattr("tris_market_event_intelligence.source_article_content.time.sleep", lambda _: None)
    store = NewsStore(tmp_path / "retry.sqlite3")
    initialize(store)
    article = Article(source="sec_8k_789019", external_id="direct", url=PRIMARY,
                      title="Disclosure", summary=None, published_at=None,
                      detected_at="2026-11-21T00:00:00+00:00")
    store.save(article.source, "https://example.com/feed", b"<rss/>", [article],
               article.detected_at, None, None)
    with store.connect() as db:
        db.execute("""
            INSERT INTO article_full_text
                (article_id,status,observed_at,error_code,retry_after)
            VALUES (1,'INCOMPLETE','2026-11-21T00:00:00+00:00',
                    'INSUFFICIENT_ARTICLE_TEXT','2099-01-01T00:00:00+00:00')
        """)
    payloads = {INDEX: TABLE, PRIMARY: BODY, EXHIBIT: RELEASE}
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=payloads[str(request.url)])
    )) as client:
        result = collect_full_text(store.path, client=client, sec_only=True)
    assert result["complete"] == 1
    assert result["missing_or_failed"] == 0


def test_incomplete_exhibit_preserves_original_and_blocks_analysis(monkeypatch):
    monkeypatch.setattr(
        "tris_market_event_intelligence.source_article_content.time.sleep", lambda _: None,
    )
    payloads = {INDEX: TABLE, PRIMARY: BODY, EXHIBIT: b"<html><body></body></html>"}
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=payloads[str(request.url)])
    )) as client:
        row = capture(client, INDEX)
    assert row["status"] == "COMPLETE"
    assert row["format"] == "SEC_BUNDLE_PARTIAL"
    assert row["error_code"] == "SEC_EXHIBIT_EMPTY_CONTENT"
    assert "Issuer reported" in row["body"]
    assert len(json.loads(row["raw_payload"])["documents"]) == 2


def test_sec_visible_text_excludes_hidden_inline_xbrl():
    from tris_market_event_intelligence.sec_operational_content import visible_text

    raw = (b"<html><head><title>metadata</title></head><body>"
           b"<ix:header><ix:hidden>hidden facts</ix:hidden></ix:header>"
           b"<p>Revenue increased.</p></body></html>")
    assert visible_text(raw, PRIMARY) == "Revenue increased."


def test_sec_http_error_has_document_role_and_status(monkeypatch):
    monkeypatch.setattr(
        "tris_market_event_intelligence.source_article_content.time.sleep", lambda _: None,
    )
    def handler(request):
        if str(request.url) == INDEX:
            return httpx.Response(200, content=TABLE)
        return httpx.Response(429)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        row = capture(client, INDEX)
    assert row["status"] == "REVIEW_REQUIRED"
    assert row["error_code"] == "SEC_PRIMARY_HTTP_429"


def test_sec_original_cache_retains_primary_when_exhibit_unavailable(tmp_path, monkeypatch):
    from tris_market_event_intelligence.source_article_content import initialize

    monkeypatch.setattr(
        "tris_market_event_intelligence.source_article_content.time.sleep", lambda _: None,
    )
    store = NewsStore(tmp_path / "originals.sqlite3")
    initialize(store)
    calls = []
    def handler(request):
        url = str(request.url)
        calls.append(url)
        if url == EXHIBIT:
            return httpx.Response(404)
        return httpx.Response(200, content={INDEX: TABLE, PRIMARY: BODY}[url])
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        first = capture_sec(client, INDEX, store=store)
        second = capture_sec(client, INDEX, store=store)
    assert first["status"] == second["status"] == "COMPLETE"
    assert first["format"] == "SEC_BUNDLE_PARTIAL"
    assert calls.count(INDEX) == calls.count(PRIMARY) == 1
    assert calls.count(EXHIBIT) == 2
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM sec_document_originals").fetchone()[0] == 2
