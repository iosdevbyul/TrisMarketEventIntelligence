"""Pilot reporting checks outcomes without modifying history or invoking inference."""
import sqlite3
from pathlib import Path

import pytest

from tris_market_event_intelligence.impact_pilot_report import (
    parse_event_ids,
    pilot_outcomes,
)


def setup(db_path: Path) -> None:
    with sqlite3.connect(db_path) as db:
        db.executescript("""
            CREATE TABLE economic_events (
                id INTEGER PRIMARY KEY, classification TEXT, status TEXT
            );
            CREATE TABLE impact_runs (
                event_id INTEGER, model_id TEXT, prompt_version TEXT, outcome TEXT
            );
            CREATE TABLE impact_errors (
                event_id INTEGER, model_id TEXT, prompt_version TEXT,
                last_error TEXT, attempts INTEGER
            );
            CREATE TABLE market_impacts (
                event_id INTEGER, model_id TEXT, prompt_version TEXT,
                direction TEXT, uncertainty TEXT, rationale TEXT
            );
            INSERT INTO economic_events VALUES (6,'INDUSTRY','UNVERIFIED');
            INSERT INTO economic_events VALUES (7,'EARNINGS','UNVERIFIED');
            INSERT INTO economic_events VALUES (8,'SUPPLY_CHAIN','UNVERIFIED');
            INSERT INTO impact_runs VALUES
                (6,'gemma2:latest','market-impact-v6','COMPLETED');
            INSERT INTO market_impacts VALUES
                (6,'gemma2:latest','market-impact-v6','POSITIVE','HIGH',
                '수출과 투자의 높은 증가세가 지속되고');
            INSERT INTO impact_errors VALUES
                (6,'gemma2:latest','market-impact-v7','RATIONALE_INCOMPLETE',1);
            INSERT INTO impact_runs VALUES
                (7,'gemma2:latest','market-impact-v7','NO_HYPOTHESIS');
            INSERT INTO impact_runs VALUES
                (8,'gemma2:latest','market-impact-v7','COMPLETED');
            INSERT INTO market_impacts VALUES
                (8,'gemma2:latest','market-impact-v7','UNCERTAIN','HIGH',
                 'The report is unverified and the outcome is uncertain.');
            INSERT INTO impact_errors VALUES
                (8,'another-model','market-impact-v7','HTTP_500',2);
        """)


def test_v7_selected_pilot_report_is_version_isolated_and_readonly(tmp_path: Path) -> None:
    db = tmp_path / "news.sqlite3"
    setup(db)
    before = db.read_bytes()
    result = pilot_outcomes(
        db, event_ids=(6, 7, 8), model_id="gemma2:latest",
        prompt_version="market-impact-v7",
    )
    assert result["status"] == "ok"
    assert result["outcome_counts"] == {
        "COMPLETED": 1, "FAILED_PENDING_RETRY": 1, "NO_HYPOTHESIS": 1,
    }
    assert result["last_failure_counts"] == {"RATIONALE_INCOMPLETE": 1}
    assert result["items"][0]["stored_impacts"] == 0
    assert result["items"][0]["last_error_code"] == "RATIONALE_INCOMPLETE"
    assert result["items"][1]["stored_impacts"] == 0
    assert result["items"][2]["direction_counts"] == {"UNCERTAIN": 1}
    assert result["items"][2]["uncertainty_counts"] == {"HIGH": 1}
    assert result["quality_verified"] is False
    assert db.read_bytes() == before


def test_historical_syntax_warning_is_not_retroactive_failure(tmp_path: Path) -> None:
    db = tmp_path / "news.sqlite3"
    setup(db)
    result = pilot_outcomes(
        db, event_ids=(6,), model_id="gemma2:latest",
        prompt_version="market-impact-v6",
    )
    assert result["items"][0]["outcome"] == "COMPLETED"
    assert result["items"][0]["syntax_warning_counts"] == {"RATIONALE_INCOMPLETE": 1}
    assert result["items"][0]["stored_impacts"] == 1


def test_missing_not_attempted_and_model_isolation(tmp_path: Path) -> None:
    db = tmp_path / "news.sqlite3"
    setup(db)
    result = pilot_outcomes(
        db, event_ids=(6, 8, 999), model_id="another-model",
        prompt_version="market-impact-v7",
    )
    assert result["outcome_counts"] == {
        "EVENT_NOT_FOUND": 1, "NOT_ATTEMPTED": 1, "FAILED_PENDING_RETRY": 1,
    }
    assert all(item["stored_impacts"] == 0 for item in result["items"])


@pytest.mark.parametrize(
    "ids",
    ["", "1,2,3,4", "1,1", "0", "-2", "1,,2", "1;2", "1, 2", "１２", "1e3"],
)
def test_invalid_id_string(ids: str) -> None:
    with pytest.raises(ValueError):
        parse_event_ids(ids)


def test_missing_and_incomplete_db_never_created(tmp_path: Path) -> None:
    db = tmp_path / "missing.sqlite3"
    assert pilot_outcomes(db, event_ids=(1,), model_id="model")["status"] == "database_missing"
    assert not db.exists()
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE unrelated (id INT)")
    before = db.read_bytes()
    assert pilot_outcomes(db, event_ids=(1,), model_id="model")["status"] == "not_initialized"
    assert db.read_bytes() == before
