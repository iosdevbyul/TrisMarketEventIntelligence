"""Cursor-paged persisted market-pressure results without legacy table mutation."""
from __future__ import annotations

import copy
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tris_market_event_intelligence.api import create_app
from tris_market_event_intelligence.market_pressure_results_store import (
    MarketPressureResultStore,
)


def report() -> dict:
    return {
        "schemaVersion": "market_pressure_auto_cycle_v1",
        "status": "CANDIDATE_EVENTS_REQUIRE_REVIEW",
        "asOfDateExclusive": "2026-11-18",
        "historicalPITVerified": False,
        "marketReturnsAnalyzed": False,
        "modelCalls": 3,
        "events": [{
            "groupSourceIds": ["article-1"],
            "proposalExplanation": "One timely financial event is observed",
            "analysis": {
                "schemaVersion": "multi_source_market_pressure_v2",
                "status": "CONDITIONAL_MARKET_PRESSURE_REQUIRES_REVIEW",
                "asOfDateExclusive": "2026-11-18",
                "historicalPITVerified": False,
                "semanticAccuracyVerified": False,
                "marketReturnsAnalyzed": False,
                "sourceDigestSha256": "a" * 64,
                "modelId": "gemma4:e2b",
                "promptVersion": "market-pressure-two-stage-v1",
                "impacts": [{
                    "direction": "NEGATIVE",
                    "market": "US", "sector": "TECH",
                    "strength": "UNASSESSED",
                    "evidence_excerpt": "cost increase",
                    "supportingSourceIds": ["article-1"],
                    "observedMarketReaction": False,
                    "pressureNotPriceForecast": True,
                }],
            },
        }],
    }


def test_market_pressure_store_is_opt_in_immutable_and_cursor_based(
    tmp_path: Path,
) -> None:
    path = tmp_path / "news.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE keep_me(value TEXT)")
        db.execute("INSERT INTO keep_me VALUES ('original')")
    store = MarketPressureResultStore(path)
    with pytest.raises(sqlite3.Error):
        store.list_results()
    with TestClient(create_app(path)) as client:
        assert client.get("/api/v1/market-pressure-research").status_code == 503

    store.initialize()
    first = store.save(report())
    assert first["inserted"] is True
    assert store.save(report()) == {"id": first["id"], "inserted": False}
    changed = copy.deepcopy(report())
    changed["events"][0]["analysis"]["impacts"][0]["direction"] = "POSITIVE"
    with pytest.raises(ValueError, match="IMMUTABLE"):
        store.save(changed)

    with TestClient(create_app(path)) as client:
        response = client.get("/api/v1/market-pressure-research?after_id=0&limit=1")
    assert response.status_code == 200
    body = response.json()
    assert body["schemaVersion"] == "market_pressure_research_feed_v1"
    assert body["items"][0]["id"] == first["id"]
    assert body["nextCursor"] == first["id"]
    assert body["hasMore"] is False
    assert store.list_results(after_id=first["id"])["items"] == []
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT value FROM keep_me").fetchone()[0] == "original"


def test_rejects_invented_market_reaction_and_corrupt_records(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE keep_me(value TEXT)")
    store = MarketPressureResultStore(path)
    store.initialize()
    tampered = report()
    tampered["events"][0]["analysis"]["impacts"][0]["observedMarketReaction"] = True
    with pytest.raises(ValueError, match="PROVENANCE"):
        store.save(tampered)
    store.save(report())
    with sqlite3.connect(path) as db:
        db.execute("UPDATE market_pressure_research SET report_json='{}'")
    with pytest.raises(ValueError, match="CHECKSUM"):
        store.list_results()
