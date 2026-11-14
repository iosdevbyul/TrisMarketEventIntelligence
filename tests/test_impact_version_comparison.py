"""Status comparisons are read-only and never certify financial accuracy."""
import sqlite3
from pathlib import Path

import pytest

from tris_market_event_intelligence.impact_version_comparison import (
    compare_impact_versions,
)


def fixture(db: Path) -> None:
    with sqlite3.connect(db) as conn:
        conn.executescript("""
            CREATE TABLE economic_events (
                id INTEGER PRIMARY KEY, classification TEXT, status TEXT
            );
            CREATE TABLE event_metadata (event_id INTEGER, claim_kind TEXT);
            CREATE TABLE impact_runs (
                event_id INTEGER, model_id TEXT, prompt_version TEXT, outcome TEXT
            );
            CREATE TABLE impact_errors (
                event_id INTEGER, model_id TEXT, prompt_version TEXT, last_error TEXT
            );
            CREATE TABLE market_impacts (
                event_id INTEGER, model_id TEXT, prompt_version TEXT, rationale TEXT
            );
            INSERT INTO economic_events VALUES (13,'EARNINGS','UNVERIFIED');
            INSERT INTO economic_events VALUES (6,'SUPPLY_CHAIN','UNVERIFIED');
            INSERT INTO economic_events VALUES (7,'ECONOMIC_INDICATOR','UNVERIFIED');
            INSERT INTO economic_events VALUES (99,'MONETARY_POLICY','CANDIDATE');
            INSERT INTO event_metadata VALUES (13,'REPORTED');
            INSERT INTO event_metadata VALUES (6,'REPORTED');
            INSERT INTO event_metadata VALUES (7,'REPORTED');
            INSERT INTO event_metadata VALUES (99,'REPORTED');
            INSERT INTO impact_runs VALUES
                (13,'gemma2:latest','market-impact-v7','COMPLETED');
            INSERT INTO impact_runs VALUES
                (13,'gemma2:latest','market-impact-v8','NO_HYPOTHESIS');
            INSERT INTO impact_runs VALUES
                (7,'gemma2:latest','market-impact-v7','NO_HYPOTHESIS');
            INSERT INTO impact_errors VALUES
                (6,'gemma2:latest','market-impact-v7','RATIONALE_INCOMPLETE');
            INSERT INTO impact_runs VALUES
                (6,'other','market-impact-v8','COMPLETED');
            INSERT INTO market_impacts VALUES
                (13,'gemma2:latest','market-impact-v7',
                'Improved performance may indicate broader economic strength.');
            INSERT INTO market_impacts VALUES
                (99,'gemma2:latest','market-impact-v7','Ineligible rumor');
        """)


def test_distinct_eligible_events_and_versioned_transitions_read_only(tmp_path: Path) -> None:
    db = tmp_path / "research.sqlite"
    fixture(db)
    before = db.read_bytes()
    report = compare_impact_versions(db, model_id="gemma2:latest")
    assert report["status"] == "ok"
    assert report["eligible_events"] == 3
    assert report["transition_counts"] == {
        "COMPLETED -> NO_HYPOTHESIS": 1,
        "FAILED_PENDING_RETRY -> NOT_ATTEMPTED": 1,
        "NO_HYPOTHESIS -> NOT_ATTEMPTED": 1,
    }
    assert report["quality_verified"] is False
    assert report["items"][0]["event_id"] == 6
    assert report["items"][0]["baseline_last_error"] == "RATIONALE_INCOMPLETE"
    assert report["items"][1]["event_id"] == 7
    assert report["items"][2]["event_id"] == 13
    assert report["items"][2]["baseline_impacts"] == 1
    assert report["items"][2]["candidate_impacts"] == 0
    assert report["items"][2]["baseline_last_error"] is None
    assert db.read_bytes() == before


def test_missing_db_and_missing_schema_are_safe(tmp_path: Path) -> None:
    path = tmp_path / "missing.sqlite3"
    report = compare_impact_versions(path, model_id="gemma2:latest")
    assert report["status"] == "database_missing"
    assert not path.exists()
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE unrelated(id INT)")
    before = path.read_bytes()
    assert compare_impact_versions(path, model_id="gemma2:latest")["status"] == "not_initialized"
    assert path.read_bytes() == before


def test_version_and_model_validation(tmp_path: Path) -> None:
    path = tmp_path / "some.sqlite"
    for kwargs in [
        {"model_id": ""},
        {"model_id": "local", "baseline": "v8", "candidate": "v8"},
        {"model_id": "local", "baseline": ""},
    ]:
        with pytest.raises(ValueError):
            compare_impact_versions(path, **kwargs)


def test_other_model_checkpoints_do_not_leak_into_comparison(tmp_path: Path) -> None:
    path = tmp_path / "some.sqlite"
    fixture(path)
    report = compare_impact_versions(path, model_id="other")
    assert report["outcome_counts"]["market-impact-v7"] == {"NOT_ATTEMPTED": 3}
    assert report["outcome_counts"]["market-impact-v8"] == {
        "COMPLETED": 1, "NOT_ATTEMPTED": 2,
    }
