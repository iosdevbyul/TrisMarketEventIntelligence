"""Collected multi-claim data enters the legacy impact processor without certification."""

import sqlite3
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from tris_market_event_intelligence.api import create_app
from tris_market_event_intelligence.claim_event_bridge import ClaimEventBridge
from tris_market_event_intelligence.claim_store import ClaimProcessor
from tris_market_event_intelligence.collector import NewsStore
from tris_market_event_intelligence.impact import EventInput, ImpactResult
from tris_market_event_intelligence.impact_store import ImpactProcessor
from tris_market_event_intelligence.operations import run_cycle

FEED = b"""<rss><channel><item><guid>case</guid>
<title>Consumer Price Index increased. CEO resignation announced.</title>
<link>https://example.org/1</link>
<description>Inflation increased. CEO resignation announced.</description>
</item></channel></rss>"""


class StubModel:
    model_id = "bridge-offline-model"

    def __init__(self) -> None:
        self.events = []

    def analyze(self, event: EventInput) -> ImpactResult:
        self.events.append(event)
        return ImpactResult(impacts=[])


def test_claims_bridge_to_analysis_preserves_provenance_and_replays(tmp_path: Path) -> None:
    db = tmp_path / "news.db"
    with httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=FEED)
    )) as client:
        result = run_cycle(
            db, selected_sources=("bbc_business",), client=client
        )
    assert result["status"] == "healthy"
    assert result["new_claims"] >= 2
    bridge = ClaimEventBridge(db)
    first = bridge.process_pending()
    assert first.created >= 2
    assert bridge.process_pending().processed == 0
    with NewsStore(db).connect() as conn:
        claims = conn.execute(
            "SELECT id, status, evidence FROM article_event_claims ORDER BY id"
        ).fetchall()
        linked = conn.execute(
            """SELECT l.claim_id, e.status, e.evidence, m.claim_kind
               FROM claim_event_links l
               JOIN economic_events e ON e.id=l.event_id
               JOIN event_metadata m ON m.event_id=e.id ORDER BY l.claim_id"""
        ).fetchall()
    assert len(linked) == len(claims)
    assert [(row[1], row[2]) for row in linked] == [
        (row[1], row[2]) for row in claims
    ]
    assert all(row[1] != "VERIFIED" for row in linked)
    with TestClient(create_app(db)) as api:
        response = api.get("/api/v1/events")
        assert response.status_code == 200
        assert len(response.json()["items"]) == first.created

    model = StubModel()
    stats = ImpactProcessor(db, model).process_pending()
    assert stats.failed == 0
    assert stats.processed >= 1
    assert all(event.status != "VERIFIED" for event in model.events)
    assert ImpactProcessor(db, model).process_pending().processed == 0


def test_forecast_claim_remains_forecast_and_is_not_analyzed(tmp_path: Path) -> None:
    db = tmp_path / "news.db"
    ClaimProcessor(db).initialize()
    with sqlite3.connect(db) as conn:
        conn.execute(
            """INSERT INTO captures
            (source, feed_url, fetched_at, sha256, payload)
            VALUES ('bok_all', 'https://example.org/feed', 'now', 'hash', X'30')"""
        )
        conn.execute(
            """INSERT INTO articles
            (source, external_id, url, title, detected_at, capture_id)
            VALUES ('bok_all', 'x', 'https://example.org/x', 'Inflation forecast', 'now', 1)"""
        )
        conn.execute(
            """INSERT INTO article_event_claims
            (article_id, extractor_version, ordinal, classification, tag, claim_kind,
             polarity, temporal_context, status, evidence, verification_note,
             source, source_url, detected_at, processed_at, capture_sha256)
            VALUES (1, 'claims-v1', 0, 'ECONOMIC_INDICATOR', 'INFLATION', 'FORECAST',
                    'AFFIRMED', 'UNSPECIFIED', 'UNVERIFIED', 'Inflation may rise',
                    'THIRD_PARTY_REPORT', 'bok_all', 'https://example.org/x',
                    'now', 'now', 'hash')"""
        )
    bridge = ClaimEventBridge(db)
    assert bridge.process_pending().created == 1
    model = StubModel()
    assert ImpactProcessor(db, model).process_pending().processed == 0
    assert not model.events
    with NewsStore(db).connect() as conn:
        claim_kind = conn.execute(
            "SELECT claim_kind FROM event_metadata"
        ).fetchone()[0]
    assert claim_kind == "FORECAST"
