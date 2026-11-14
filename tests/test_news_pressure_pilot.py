"""Read-only pressure pilot safeguards with locally seeded SQLite data."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tris_market_event_intelligence.impact import ImpactAnalysisError
from tris_market_event_intelligence.news_impact_signal import NewsImpactSignal
from tris_market_event_intelligence.news_pressure_pilot import (
    run_pressure_pilot,
    select_pressure_pilot,
)

EVIDENCE = "Federal funds rate The Committee raised the interest rate by 25 basis points."
RATIONALE = (
    "Higher rates may pressure technology valuations via discount rates, "
    "but market expectations and repricing are unknown."
)


def fixture_db(path: Path) -> None:
    with sqlite3.connect(path) as db:
        db.executescript(
            """CREATE TABLE economic_events (
                id INTEGER PRIMARY KEY, classification TEXT, status TEXT,
                direction TEXT, change_basis_points INTEGER, evidence TEXT,
                source TEXT, source_url TEXT, detected_at TEXT
            );
            CREATE TABLE event_metadata (
                event_id INTEGER, claim_kind TEXT, verification_note TEXT,
                tags_json TEXT
            );
            CREATE TABLE market_impacts (
                id INTEGER PRIMARY KEY, event_id INTEGER, market TEXT,
                sector TEXT, direction TEXT, horizon TEXT, rationale TEXT,
                evidence_excerpt TEXT, uncertainty TEXT, drivers_json TEXT,
                entities_json TEXT
            );"""
        )
        db.execute(
            "INSERT INTO economic_events VALUES (?,?,?,?,?,?,?,?,?)",
            (17, "MONETARY_POLICY", "VERIFIED", "UP", 25, EVIDENCE,
             "fed_monetary", "https://www.federalreserve.gov/a",
             "2026-11-15T00:00:00+00:00"),
        )
        db.execute(
            "INSERT INTO event_metadata VALUES (?,?,?,?)",
            (17, "REPORTED", "Policy source rule", "[]"),
        )
        db.execute(
            "INSERT INTO market_impacts VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (1, 17, "US", "TECH", "NEGATIVE", "SHORT_TERM",
             RATIONALE, "raised the interest rate by 25 basis points",
             "HIGH", '["discount rates"]', "[]"),
        )


class MockPressure:
    model = "mock-gemma2"

    def __init__(self, *, error: bool = False) -> None:
        self.calls = 0
        self.error = error

    def analyze(self, event, impact, *, generated_at):
        self.calls += 1
        if self.error:
            raise ImpactAnalysisError("NEWS_INTENSITY_UNSUPPORTED", "Rejected evidence")
        return NewsImpactSignal(
            event_id=event.event_id, market=impact.market, sector=impact.sector,
            direction=impact.direction, intensity="MODERATE",
            intensity_score=45.0, horizon=impact.horizon,
            uncertainty=impact.uncertainty, rationale=impact.rationale,
            evidence_excerpt=impact.evidence_excerpt, source_url=event.source_url,
            model_id=self.model, prompt_version="news-pressure-intensity-v1",
            generated_at=generated_at,
        )


def test_pilot_previews_without_calls_or_db_writes(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    fixture_db(path)
    with sqlite3.connect(path) as db:
        before = tuple(db.iterdump())
    records = select_pressure_pilot(path)
    assert [(i, e.event_id, p.direction) for i, e, p in records] == [
        (1, 17, "NEGATIVE"),
    ]
    preview = run_pressure_pilot(path, analyzer=None)
    assert preview["status"] == "PREVIEW_ONLY"
    assert preview["selected"] == 1
    assert preview["items"][0]["status"] == "PREVIEW_NOT_ANALYZED"
    assert preview["production_database_modified"] is False
    assert preview["new_pressure_signals_persisted"] == 0
    with sqlite3.connect(path) as db:
        assert tuple(db.iterdump()) == before


def test_pilot_model_output_and_rejection_are_isolated(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    fixture_db(path)
    with sqlite3.connect(path) as db:
        before = tuple(db.iterdump())
    analyzer = MockPressure()
    report = run_pressure_pilot(path, analyzer=analyzer)
    assert analyzer.calls == 1
    assert report["status"] == "PILOT_OUTPUT_VALIDATED"
    assert report["items"][0]["intensity_score"] == 45
    assert report["items"][0]["signal"]["predicts_stock_return"] is False
    assert report["live_model_quality_verified"] is False
    rejected = run_pressure_pilot(path, analyzer=MockPressure(error=True))
    assert rejected["status"] == "PILOT_COMPLETED_WITH_REJECTIONS"
    assert rejected["items"][0]["failure_code"] == "NEWS_INTENSITY_UNSUPPORTED"
    with sqlite3.connect(path) as db:
        assert tuple(db.iterdump()) == before


def test_empty_and_missing_db_fail_without_fake_model_results(tmp_path: Path) -> None:
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(ValueError, match="Existing"):
        run_pressure_pilot(missing, analyzer=MockPressure())
    assert not missing.exists()
    empty = tmp_path / "empty.sqlite3"
    fixture_db(empty)
    with sqlite3.connect(empty) as db:
        db.execute("DELETE FROM market_impacts")
    report = run_pressure_pilot(empty, analyzer=MockPressure())
    assert report["status"] == "NO_EXISTING_ELIGIBLE_IMPACTS"
    assert report["selected"] == 0


def test_out_of_scope_event_and_invalid_pilot_limit_are_rejected(
    tmp_path: Path,
) -> None:
    path = tmp_path / "news.sqlite3"
    fixture_db(path)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE economic_events SET status='CANDIDATE' WHERE id=17")
    assert select_pressure_pilot(path) == []
    with pytest.raises(ValueError, match="1..3"):
        select_pressure_pilot(path, limit=4)
