import hashlib
import sqlite3
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from tris_market_event_intelligence.collector import (
    NewsStore,
    OfficialNewsCollector,
    parse_rss,
)

RSS = b"""<?xml version="1.0"?>
<rss version="2.0"><channel><item>
<title>Policy statement</title>
<link>https://example.org/a</link>
<guid>statement-001</guid>
<pubDate>Fri, 13 Nov 2026 18:00:00 GMT</pubDate>
<description>Official press release</description>
</item></channel></rss>"""


def test_parse_rss_preserves_publication_and_detection_times() -> None:
    items = parse_rss("fed", RSS, "2026-11-14T00:00:00+00:00")
    assert len(items) == 1
    assert items[0].external_id == "statement-001"
    assert items[0].published_at == "2026-11-13T18:00:00+00:00"
    assert items[0].detected_at == "2026-11-14T00:00:00+00:00"


def test_invalid_xml_and_excessive_payload_fail() -> None:
    with pytest.raises(ET.ParseError):
        parse_rss("fed", b"<rss>", "now")
    with pytest.raises(ValueError, match="maximum size"):
        parse_rss("fed", b"x" * 2_000_001, "now")


def test_repeated_fetch_is_idempotent_and_retains_raw_evidence(tmp_path: Path) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.headers.get("if-none-match") == '"v1"':
            return httpx.Response(304)
        return httpx.Response(200, content=RSS, headers={"ETag": '"v1"'})

    store = NewsStore(tmp_path / "news.sqlite3")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    collector = OfficialNewsCollector(store, client, {"fed": "https://example.org/feed"})
    assert collector.collect_once()["fed"] == {"status": "ok", "inserted": 1}
    assert collector.collect_once()["fed"] == {"status": "not_modified", "inserted": 0}
    assert store.stats() == {"articles": 1, "captures": 1}
    assert calls[1].headers["if-none-match"] == '"v1"'
    with store.connect() as db:
        digest, payload = db.execute("SELECT sha256, payload FROM captures").fetchone()
    assert payload == RSS
    assert digest == hashlib.sha256(RSS).hexdigest()


def test_same_payload_without_etag_is_idempotent(tmp_path: Path) -> None:
    store = NewsStore(tmp_path / "news.sqlite3")
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=RSS))
    )
    collector = OfficialNewsCollector(store, client, {"fed": "https://example.org/feed"})
    assert collector.collect_once()["fed"]["inserted"] == 1
    assert collector.collect_once()["fed"]["inserted"] == 0
    assert store.stats() == {"articles": 1, "captures": 1}


def test_error_is_recorded_and_next_poll_recovers(tmp_path: Path) -> None:
    attempts = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503)
        return httpx.Response(200, content=RSS)

    store = NewsStore(tmp_path / "news.sqlite3")
    client = httpx.Client(transport=httpx.MockTransport(handler))
    collector = OfficialNewsCollector(store, client, {"fed": "https://example.org/feed"})
    assert collector.collect_once()["fed"]["status"] == "error"
    with store.connect() as db:
        assert db.execute("SELECT last_error FROM fetch_state").fetchone()[0]
    assert collector.collect_once()["fed"]["status"] == "ok"
    assert store.stats()["articles"] == 1
    with store.connect() as db:
        assert db.execute("SELECT last_error FROM fetch_state").fetchone()[0] is None


def test_atomic_save_rolls_back_on_invalid_article(tmp_path: Path) -> None:
    store = NewsStore(tmp_path / "news.sqlite3")
    store.initialize()
    with store.connect() as db:
        db.execute(
            """CREATE TRIGGER reject_bad_article BEFORE INSERT ON articles
            WHEN NEW.external_id = 'bad' BEGIN SELECT RAISE(ABORT, 'bad article'); END"""
        )
    with pytest.raises(sqlite3.IntegrityError, match="bad article"):
        store.save(
            "fed", "https://example.org/feed", RSS,
            [parse_rss("fed", RSS, "now")[0], replace(parse_rss("fed", RSS, "now")[0], external_id="bad", url="https://example.org/bad")],
            "now", None, None,
        )
    assert store.stats() == {"articles": 0, "captures": 0}


def test_atom_support() -> None:
    atom = b"""<feed xmlns="http://www.w3.org/2005/Atom"><entry>
    <title>New decision</title><id>abc</id>
    <link href="https://example.org/decision"/>
    <updated>2026-11-14T01:00:00Z</updated></entry></feed>"""
    articles = parse_rss("bok", atom, "2026-11-14T02:00:00Z")
    assert len(articles) == 1
    assert articles[0].published_at == "2026-11-14T01:00:00+00:00"


def test_rotating_rss_guids_for_same_source_url_are_idempotent(tmp_path: Path) -> None:
    store = NewsStore(tmp_path / "news.sqlite3")
    initial = parse_rss("bbc_business", RSS, "2026-11-14T00:00:00+00:00")[0]
    second = replace(
        initial, external_id="new-guid-from-next-poll",
        title="Rewritten title from the same URL",
        detected_at="2026-11-15T02:00:00+00:00",
    )
    store.initialize()
    assert store.save("bbc_business", "https://example.org/rss", RSS, [initial],
                      initial.detected_at, None, None) == 1
    assert store.save("bbc_business", "https://example.org/rss", b"updated feed",
                      [second], second.detected_at, None, None) == 0
    with store.connect() as db:
        articles = db.execute(
            "SELECT external_id, title, detected_at FROM articles"
        ).fetchall()
    assert articles == [(
        initial.external_id, initial.title, initial.detected_at,
    )]
    assert store.stats() == {"articles": 1, "captures": 2}


def test_exact_url_dedup_preserves_source_boundaries(tmp_path: Path) -> None:
    store = NewsStore(tmp_path / "news.sqlite3")
    article = parse_rss("bbc_business", RSS, "2026-11-14T00:00:00+00:00")[0]
    store.initialize()
    assert store.save("bbc_business", "https://example.org/feed", RSS, [article],
                      article.detected_at, None, None) == 1
    distinct = replace(article, source="bbc_world")
    assert store.save("bbc_world", "https://example.org/feed", RSS, [distinct],
                      distinct.detected_at, None, None) == 1
    assert store.stats()["articles"] == 2
