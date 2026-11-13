"""Read-only coverage of eligible real events and version-isolated model errors."""
import sqlite3
from pathlib import Path

import pytest

from tris_market_event_intelligence.impact_coverage import impact_coverage


def prepare(db: Path) -> None:
    with sqlite3.connect(db) as connection:
        connection.executescript("""
            CREATE TABLE economic_events (
                id INTEGER PRIMARY KEY, classification TEXT, status TEXT
            );
            CREATE TABLE event_metadata (event_id INTEGER, claim_kind TEXT);
            CREATE TABLE impact_runs (
                event_id INTEGER, model_id TEXT, prompt_version TEXT, outcome TEXT
            );
            CREATE TABLE market_impacts (
                event_id INTEGER, model_id TEXT, prompt_version TEXT,
                direction TEXT, uncertainty TEXT
            );
            CREATE TABLE impact_errors (
                event_id INTEGER, model_id TEXT, prompt_version TEXT,
                last_error TEXT, attempts INTEGER
            );
        """)
        connection.executemany(
            "INSERT INTO economic_events VALUES (?,?,?)", [
                (1, "INDUSTRY", "UNVERIFIED"),
                (2, "INDUSTRY", "UNVERIFIED"),
                (3, "MONETARY_POLICY", "VERIFIED"),
                (4, "MONETARY_POLICY", "CANDIDATE"),
                (5, "EARNINGS", "UNVERIFIED"),
                (6, "EARNINGS", "UNVERIFIED"),
            ],
        )
        connection.executemany(
            "INSERT INTO event_metadata VALUES (?,?)", [
                (1, "REPORTED"), (2, "REPORTED"), (3, "REPORTED"),
                (4, "REPORTED"), (5, "FORECAST"), (6, "REPORTED"),
            ],
        )
        connection.executemany(
            "INSERT INTO impact_runs VALUES (?,?,?,?)", [
                (1, "gemma2:latest", "market-impact-v6", "COMPLETED"),
                (2, "gemma2:latest", "market-impact-v7", "NO_HYPOTHESIS"),
                (3, "gemma2:latest", "market-impact-v7", "COMPLETED"),
                (6, "other-model", "market-impact-v7", "COMPLETED"),
            ],
        )
        connection.execute(
            "INSERT INTO market_impacts VALUES (3,'gemma2:latest',"
            "'market-impact-v7','UNCERTAIN','HIGH')"
        )
        connection.executemany(
            "INSERT INTO impact_errors VALUES (?,?,?,?,?)", [
                (1, "gemma2:latest", "market-impact-v7", "RATIONALE_INCOMPLETE", 2),
                (4, "gemma2:latest", "market-impact-v7", "IGNORED", 1),
            ],
        )


def test_versioned_census_is_read_only_and_distinct(tmp_path: Path) -> None:
    db = tmp_path / "news.sqlite3"
    prepare(db)
    before = db.read_bytes()
    report = impact_coverage(db, model_id="gemma2:latest")
    assert report["total_events"] == 6
    assert report["eligible_events"] == 4
    assert report["ineligible_events"] == 2
    assert report["completed_events"] == 1
    assert report["no_hypothesis_events"] == 1
    assert report["pending_events"] == 2
    assert report["pending_sample_event_ids"] == [1, 6]
    assert report["pending_with_recorded_failure"] == 1
    assert report["last_failure_counts"] == {"RATIONALE_INCOMPLETE": 1}
    assert report["stored_impacts"] == 1
    assert report["direction_counts"] == {"UNCERTAIN": 1}
    assert report["historical_run_counts"] == [
        {"prompt_version": "market-impact-v6", "outcome": "COMPLETED", "n": 1},
        {"prompt_version": "market-impact-v7", "outcome": "COMPLETED", "n": 1},
        {"prompt_version": "market-impact-v7", "outcome": "NO_HYPOTHESIS", "n": 1},
    ]
    assert report["quality_verified"] is False
    assert db.read_bytes() == before


def test_missing_database_and_invalid_limits(tmp_path: Path) -> None:
    missing = tmp_path / "missing.sqlite3"
    assert impact_coverage(missing, model_id="gemma2:latest")["status"] == "database_missing"
    assert not missing.exists()
    with pytest.raises(ValueError, match="sample_limit"):
        impact_coverage(missing, model_id="gemma2:latest", sample_limit=0)


def test_missing_tables_are_reported_not_created(tmp_path: Path) -> None:
    db = tmp_path / "partial.sqlite3"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE unrelated (id INTEGER)")
    before = db.read_bytes()
    assert impact_coverage(db, model_id="gemma2:latest")["status"] == "not_initialized"
    assert db.read_bytes() == before
