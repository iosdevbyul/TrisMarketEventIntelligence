"""Offline HTTP E2E: event pagination and delayed impact processing."""

from pathlib import Path

from fastapi.testclient import TestClient

from tris_market_event_intelligence.api import create_app
from tris_market_event_intelligence.collector import NewsStore, parse_rss
from tris_market_event_intelligence.event_store import EventProcessor
from tris_market_event_intelligence.impact import Impact, ImpactResult
from tris_market_event_intelligence.impact_store import ImpactProcessor


class Provider:
    model_id = "integration-test"

    def analyze(self, event):
        return ImpactResult(impacts=[Impact(
            market="US", sector="TECH", direction="UNCERTAIN",
            horizon="SHORT_TERM",
            rationale="Higher policy rates can weigh on growth valuations if unexpected.",
            evidence_excerpt="raised the interest rate by 25 basis points",
            uncertainty="HIGH", drivers=["discount rate", "expectations unknown"],
        )])


def populate(path: Path) -> None:
    store = NewsStore(path)
    store.initialize()
    payload = b"""<rss><channel>
    <item><title>Federal funds rate</title><link>https://example.org/one</link>
    <guid>one</guid><description>The Committee raised the interest rate by 25 basis points.</description></item>
    <item><title>Federal funds rate</title><link>https://example.org/two</link>
    <guid>two</guid><description>The Committee may raise the interest rate by 25 basis points.</description></item>
    <item><title>Holiday</title><link>https://example.org/three</link>
    <guid>three</guid><description>Office hours</description></item>
    </channel></rss>"""
    articles = parse_rss("fed_monetary", payload, "2026-11-14T00:00:00+00:00")
    store.save("fed_monetary", "https://example.org/feed", payload, articles, "now", None, None)
    EventProcessor(path).process_pending()
    # Creates the tables but no AI analysis yet.
    ImpactProcessor(path, Provider()).initialize()


def test_api_health_and_readiness(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    with TestClient(create_app(path)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.get("/api/v1/events").status_code == 503
        populate(path)
        assert client.get("/ready").json() == {"status": "ready"}


def test_cursor_pagination_and_filtering(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    populate(path)
    with TestClient(create_app(path)) as client:
        first = client.get("/api/v1/events", params={"limit": 1}).json()
        assert len(first["items"]) == 1
        assert first["hasMore"] is True
        assert first["items"][0]["status"] == "VERIFIED"
        cursor = first["nextCursor"]
        second = client.get("/api/v1/events", params={"after_id": cursor}).json()
        assert len(second["items"]) == 1
        assert second["items"][0]["status"] == "CANDIDATE"
        assert second["nextCursor"] > cursor
        assert client.get("/api/v1/events", params={"after_id": second["nextCursor"]}).json()[
            "items"
        ] == []
        filtered = client.get("/api/v1/events", params={"status": "VERIFIED"}).json()
        assert len(filtered["items"]) == 1
        assert client.get("/api/v1/events", params={"status": "INVALID"}).status_code == 422
        assert client.get("/api/v1/events", params={"limit": 501}).status_code == 422
        assert client.get("/api/v1/events", params={"after_id": -1}).status_code == 422


def test_delayed_analysis_has_separate_cursor(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    populate(path)
    with TestClient(create_app(path)) as client:
        initial = client.get("/api/v1/events").json()
        latest_event_cursor = initial["nextCursor"]
        assert latest_event_cursor > 0
        assert client.get("/api/v1/impacts").json()["items"] == []
        assert client.get("/api/v1/analysis-runs").json()["items"] == []

        # AI analysis is generated after the event cursor has already advanced.
        assert ImpactProcessor(path, Provider()).process_pending().processed == 1
        assert client.get("/api/v1/events", params={"after_id": latest_event_cursor}).json()[
            "items"
        ] == []
        impacts = client.get("/api/v1/impacts", params={"limit": 1}).json()
        assert len(impacts["items"]) == 1
        assert impacts["items"][0]["drivers"] == ["discount rate", "expectations unknown"]
        assert impacts["items"][0]["evidence_excerpt"]
        assert client.get("/api/v1/impacts", params={
            "after_id": impacts["nextCursor"]
        }).json()["items"] == []
        runs = client.get("/api/v1/analysis-runs").json()
        assert runs["items"][0]["outcome"] == "COMPLETED"


def test_read_only_api_does_not_generate_events(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    populate(path)
    with TestClient(create_app(path)) as client:
        for _ in range(3):
            assert client.get("/api/v1/events").status_code == 200
            assert client.get("/api/v1/impacts").status_code == 200
            assert client.get("/api/v1/analysis-runs").status_code == 200
    with NewsStore(path).connect() as db:
        assert db.execute("SELECT count(*) FROM economic_events").fetchone()[0] == 2
        assert db.execute("SELECT count(*) FROM market_impacts").fetchone()[0] == 0
