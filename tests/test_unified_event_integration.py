import hashlib
import json
import sqlite3
from datetime import UTC, datetime

import pytest

from tris_market_event_intelligence.economic_collection_cli import SCHEMA, save
from tris_market_event_intelligence.unified_event_integration_cli import integrate, links

STAMP = "2026-01-01T12:00:00+00:00"
CUTOFF = datetime(2026, 1, 2, tzinfo=UTC)


def test_readonly_snapshot_and_candidates(tmp_path):
    news, macro, out = (tmp_path / p for p in ("news.db", "macro.db", "events.db"))
    raw = b"original feed"
    sha = hashlib.sha256(raw).hexdigest()
    with sqlite3.connect(news) as db:
        db.executescript("""
            CREATE TABLE captures(id INTEGER PRIMARY KEY,sha256,payload,fetched_at);
            CREATE TABLE articles(id INTEGER PRIMARY KEY,source,external_id,url,title,summary,
                                  published_at,detected_at,capture_id);
            CREATE TABLE article_full_text(article_id,status,observed_at,body,body_sha256,
                                           raw_sha256,raw_payload,sparse_pages);
        """)
        db.execute("INSERT INTO captures VALUES (1,?,?,?)", (sha, raw, STAMP))
        title = "NVIDIA quarterly results beat revenue estimates today"
        db.executemany("INSERT INTO articles VALUES (?,?,?,?,?,?,?,?,1)", [
            (1, "bbc_business", "1", "https://bbc.com/1", title, None, STAMP, STAMP),
            (2, "cnbc_top", "2", "https://cnbc.com/2", title, None, STAMP, STAMP),
            (3, "sec_filing_1045810", "0001045810-26-000001",
             ("https://www.sec.gov/Archives/edgar/data/1045810/000104581026000001/"
              "0001045810-26-000001-index.htm"), "NVIDIA quarterly results",
             '{"form":"8-K"}', STAMP, STAMP),
            (4, "bea_news", "4", "https://bea.gov/news/4", "GDP release",
             None, None, STAMP),
            (5, "bbc_business", "5", "https://bbc.com/5", "Future article", None,
             "2026-01-03T00:00:00+00:00", "2026-01-03T00:00:00+00:00"),
        ])
        body = "NVIDIA announced quarterly results."
        db.executemany("INSERT INTO article_full_text VALUES (?,?,?,?,?,?,?,?)", [
            (i, "COMPLETE", STAMP, body, hashlib.sha256(body.encode()).hexdigest(),
             sha, raw, "[]") for i in (1, 2, 3)
        ])
    with sqlite3.connect(macro) as db:
        db.executescript(SCHEMA)
        save(db, "FRED_CSV", "CPIAUCSL", {"values": ["100"]},
             [("2025-12-01", "", "", "100")])
        db.execute("UPDATE macro_observation SET fetched_at=?", (STAMP,))
        db.execute("UPDATE macro_raw SET fetched_at=?", (STAMP,))
    before = (news.read_bytes(), macro.read_bytes())
    report = integrate(news, macro, out, cutoff=CUTOFF)
    assert report["byKind"] == {"NEWS": 2, "SEC": 1, "ECONOMIC_RELEASE": 1,
                                "ECONOMIC_OBSERVATION": 1}
    assert report["predictionEligible"] == 0
    assert report["byRelationship"]["CROSS_SOURCE_EVENT_CANDIDATE"] == 2
    assert report["byRelationship"]["REPRINT_CANDIDATE"] == 1
    assert integrate(news, macro, out, cutoff=CUTOFF)["reusedSnapshot"]
    assert before == (news.read_bytes(), macro.read_bytes())
    with sqlite3.connect(out) as db:
        data = json.loads(db.execute(
            "SELECT payload FROM event_candidates WHERE kind='ECONOMIC_OBSERVATION'"
        ).fetchone()[0])
        assert data["period"] == "2025-12-01"
        assert data["firstPublicAt"] is None
        assert data["vintageStart"] is None
        assert data["observedAt"] == STAMP
    with pytest.raises(ValueError):
        integrate(news, macro, news, cutoff=CUTOFF)


def test_macro_versions_never_become_verified_releases():
    events = [
        {"eventId": str(i), "kind": "ECONOMIC_OBSERVATION", "source": "FRED",
         "series": "CPI", "period": "2025-12-01", "observedAt": STAMP,
         "quality": "NUMERIC_CONTEXT_ONLY"} for i in range(2)
    ]
    edge = next(links(events))
    assert edge[2] == "OBSERVATION_VERSION_CANDIDATE"
    assert not json.loads(edge[3])["economicEventVerified"]
