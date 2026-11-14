"""Real SQLite evidence selection with stubbed news-only 2-stage models."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tris_market_event_intelligence.impact import Impact, ImpactResult
from tris_market_event_intelligence.news_impact_signal import NewsImpactSignal
from tris_market_event_intelligence.news_two_stage_pilot import run_two_stage_pilot

EVIDENCE = "Federal funds rate The Committee raised the interest rate by 25 basis points."
RATIONALE = (
    "Higher policy rates could reduce expected tech valuations via discount "
    "rates, while market expectations and repricing remain unknown."
)


def db_seed(path: Path) -> None:
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
                id INTEGER PRIMARY KEY, event_id INTEGER
            );
            CREATE TABLE impact_runs (
                event_id INTEGER, model_id TEXT, prompt_version TEXT,
                outcome TEXT
            );
            CREATE TABLE impact_errors (
                event_id INTEGER, model_id TEXT, prompt_version TEXT,
                last_error TEXT
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
            (17, "REPORTED", "Official source", "[]"),
        )


class Direction:
    model_id = "gemma2:latest"

    def __init__(self, direction: str = "NEGATIVE", bad: bool = False) -> None:
        self.calls = 0
        self.direction = direction
        self.bad = bad

    def analyze(self, event):
        self.calls += 1
        if self.direction == "NONE":
            return ImpactResult(impacts=[])
        return ImpactResult(impacts=[Impact(
            market="US", sector="TECH", direction=self.direction,
            horizon="SHORT_TERM", rationale=RATIONALE,
            evidence_excerpt=(
                "invented source" if self.bad
                else "raised the interest rate by 25 basis points"
            ),
            uncertainty="HIGH", drivers=["discount rates"],
        )])


class Strength:
    model = "gemma2:latest"

    def __init__(self) -> None:
        self.calls = 0

    def analyze(self, event, impact, *, generated_at):
        self.calls += 1
        return NewsImpactSignal(
            event_id=event.event_id, market=impact.market, sector=impact.sector,
            direction=impact.direction, intensity="MODERATE",
            intensity_score=49.0, horizon=impact.horizon,
            uncertainty=impact.uncertainty, rationale=impact.rationale,
            evidence_excerpt=impact.evidence_excerpt,
            source_url=event.source_url, model_id=self.model,
            prompt_version="news-pressure-intensity-v1",
            generated_at=generated_at,
        )


def test_read_only_two_stage_real_event_from_sqlite(tmp_path: Path) -> None:
    path = tmp_path / "real.sqlite3"
    db_seed(path)
    with sqlite3.connect(path) as db:
        baseline = tuple(db.iterdump())
    direction, strength = Direction(), Strength()
    report = run_two_stage_pilot(
        path, direction_model=direction, strength_model=strength,
    )
    assert report["status"] == "TWO_STAGE_PILOT_FINISHED"
    assert direction.calls == 1
    assert strength.calls == 1
    assert report["direction_model_calls"] == 1
    assert report["strength_model_calls"] == 1
    assert report["items"][0]["signal"]["direction"] == "NEGATIVE"
    assert report["items"][0]["signal"]["intensity_score"] == 49
    assert report["items"][0]["signal"]["predicts_stock_price"] is False
    assert report["production_database_modified"] is False
    with sqlite3.connect(path) as db:
        assert tuple(db.iterdump()) == baseline


@pytest.mark.parametrize("direction", ["UNCERTAIN", "NONE"])
def test_unresolved_first_stage_never_calls_strength(
    tmp_path: Path, direction: str,
) -> None:
    path = tmp_path / "real.sqlite3"
    db_seed(path)
    first, second = Direction(direction), Strength()
    report = run_two_stage_pilot(
        path, direction_model=first, strength_model=second,
    )
    assert first.calls == 1
    assert second.calls == 0
    assert report["strength_model_calls"] == 0
    assert report["items"][0]["status"] == "NO_SUPPORTED_DIRECTION"


def test_invalid_first_stage_source_quote_prevents_strength(
    tmp_path: Path,
) -> None:
    path = tmp_path / "real.sqlite3"
    db_seed(path)
    first, second = Direction(bad=True), Strength()
    report = run_two_stage_pilot(
        path, direction_model=first, strength_model=second,
    )
    assert first.calls == 1
    assert second.calls == 0
    assert report["items"][0]["status"] == "REJECTED_OR_FAILED"
    assert report["items"][0]["failure_code"] == "EVIDENCE_MISMATCH"


def test_preview_not_checkpointed_and_existing_processed_event_excluded(
    tmp_path: Path,
) -> None:
    path = tmp_path / "real.sqlite3"
    db_seed(path)
    preview = run_two_stage_pilot(
        path, direction_model=None, strength_model=None,
    )
    assert preview["selected_events"] == 1
    assert preview["direction_model_calls"] == 0
    assert preview["status"] == "PREVIEW_ONLY"
    with sqlite3.connect(path) as db:
        db.execute(
            "INSERT INTO impact_runs VALUES (?,?,?,?)",
            (17, "gemma2:latest", "market-impact-v8", "COMPLETED"),
        )
    first, second = Direction(), Strength()
    report = run_two_stage_pilot(
        path, direction_model=first, strength_model=second,
    )
    assert report["status"] == "NO_ELIGIBLE_NEW_EVENTS"
    assert first.calls == second.calls == 0


def test_missing_database_and_one_sided_model_configuration_fail(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing.sqlite3"
    with pytest.raises(ValueError, match="Existing"):
        run_two_stage_pilot(
            path, direction_model=Direction(), strength_model=Strength(),
        )
    assert not path.exists()
    with pytest.raises(ValueError, match="together"):
        run_two_stage_pilot(
            path, direction_model=Direction(), strength_model=None,
        )
