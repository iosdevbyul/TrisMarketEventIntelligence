import base64
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from tris_market_event_intelligence.collector import Article, NewsStore
from tris_market_event_intelligence.sec_operational_content import documents, index_identity
from tris_market_event_intelligence.source_article_content import capture, collect_full_text

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
