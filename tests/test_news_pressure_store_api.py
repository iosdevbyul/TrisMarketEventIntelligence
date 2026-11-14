"""SQLite and API behavior for the opt-in, news-only signal stream."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tris_market_event_intelligence.api import create_app
from tris_market_event_intelligence.news_impact_signal import NewsImpactSignal
from tris_market_event_intelligence.news_pressure_store import NewsPressureStore


def fixture_db(path: Path) -> None:
    with sqlite3.connect(path) as db:
        db.executescript(
            """CREATE TABLE economic_events (
                id INTEGER PRIMARY KEY, source_url TEXT NOT NULL
            );
            CREATE TABLE market_impacts (
                id INTEGER PRIMARY KEY, event_id INTEGER NOT NULL,
                market TEXT NOT NULL, sector TEXT NOT NULL,
                direction TEXT NOT NULL, horizon TEXT NOT NULL,
                uncertainty TEXT NOT NULL, evidence_excerpt TEXT NOT NULL
            );
            INSERT INTO economic_events VALUES (17, 'https://example.org/news/17');
            INSERT INTO market_impacts VALUES (
                1, 17, 'KR', 'SEMICONDUCTORS', 'NEGATIVE',
                'SHORT_TERM', 'HIGH', 'Export restrictions on semiconductors'
            );
            INSERT INTO market_impacts VALUES (
                2, 17, 'KR', 'SEMICONDUCTORS', 'NEGATIVE',
                'SHORT_TERM', 'HIGH', 'Export restrictions on semiconductors'
            );"""
        )


def signal(*, score: float = 52.0, prompt: str = "news-pressure-intensity-v1"
           ) -> NewsImpactSignal:
    return NewsImpactSignal(
        event_id=17, market="KR", sector="SEMICONDUCTORS",
        direction="NEGATIVE", intensity="MODERATE", intensity_score=score,
        horizon="SHORT_TERM", uncertainty="HIGH",
        rationale="If export limits tighten, expected chip shipments could decrease.",
        evidence_excerpt="Export restrictions on semiconductors",
        source_url="https://example.org/news/17",
        model_id="gemma2:latest", prompt_version=prompt,
        generated_at="2026-11-15T00:00:00+00:00",
    )


def test_news_pressure_store_is_explicit_and_api_is_read_only(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    fixture_db(path)
    store = NewsPressureStore(path)
    with TestClient(create_app(path)) as client:
        unavailable = client.get("/api/v1/news-impact-signals")
        assert unavailable.status_code == 503
    with sqlite3.connect(path) as db:
        assert db.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name='news_pressure_signals'"
        ).fetchone()[0] == 0
    store.initialize()
    before = store.list_signals()
    assert before["items"] == []
    result = store.save(impact_id=1, signal=signal())
    assert result["inserted"] is True
    assert result["id"] == 1
    assert store.save(impact_id=1, signal=signal()) == {
        **result, "inserted": False,
    }
    store.save(impact_id=2, signal=signal())
    with TestClient(create_app(path)) as client:
        first = client.get("/api/v1/news-impact-signals", params={"limit": 1})
        assert first.status_code == 200
        item = first.json()["items"][0]
        assert item["sourceImpactId"] == 1
        assert item["intensity"] == "MODERATE"
        assert item["intensity_score"] == 52
        assert item["price_prediction_owned_by"] == "DonghakStockVision"
        assert item["predicts_stock_return"] is False
        assert "predicted_stock_price" not in item
        assert first.json()["hasMore"] is True
        assert first.json()["nextCursor"] == 1
        second = client.get("/api/v1/news-impact-signals",
                            params={"after_id": 1, "limit": 1})
        assert second.status_code == 200
        assert second.json()["items"][0]["id"] == 2
        assert second.json()["hasMore"] is False
        assert client.get("/api/v1/news-impact-signals",
                          params={"limit": 0}).status_code == 422


def test_conflicting_signal_and_forged_source_provenance_fail_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "news.sqlite3"
    fixture_db(path)
    store = NewsPressureStore(path)
    store.initialize()
    store.save(impact_id=1, signal=signal())
    with pytest.raises(ValueError, match="Immutable"):
        store.save(impact_id=1, signal=signal(score=55))
    forged = signal().model_copy(update={"sector": "ENERGY"})
    with pytest.raises(ValueError, match="provenance"):
        store.save(impact_id=2, signal=forged)
    with pytest.raises(ValueError, match="version"):
        store.save(impact_id=2, signal=signal(prompt="legacy-v8"))
    with pytest.raises(ValueError, match="not found"):
        store.save(impact_id=99, signal=signal())
    assert len(store.list_signals()["items"]) == 1


def test_tampered_persisted_record_is_not_served_as_verified(
    tmp_path: Path,
) -> None:
    path = tmp_path / "news.sqlite3"
    fixture_db(path)
    store = NewsPressureStore(path)
    store.initialize()
    store.save(impact_id=1, signal=signal())
    with sqlite3.connect(path) as db:
        db.execute(
            "UPDATE news_pressure_signals SET payload_sha256=? WHERE id=1",
            ("a" * 64,),
        )
    with pytest.raises(ValueError, match="integrity"):
        store.list_signals()
    with TestClient(create_app(path)) as client:
        assert client.get("/api/v1/news-impact-signals").status_code == 503


def test_missing_database_is_not_created_and_has_no_side_effects(tmp_path: Path) -> None:
    path = tmp_path / "missing.sqlite3"
    store = NewsPressureStore(path)
    with pytest.raises(ValueError, match="Existing"):
        store.initialize()
    assert not path.exists()
    with TestClient(create_app(path)) as client:
        assert client.get("/api/v1/news-impact-signals").status_code == 503
    assert not path.exists()
