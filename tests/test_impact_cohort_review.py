"""Private source-linked review for never-before-analyzed economic events."""
import json
import os
import sqlite3
import stat
from pathlib import Path

import pytest

from tris_market_event_intelligence.impact_cohort_review import (
    cohort_review,
    save_cohort_review,
)


def fixture(path: Path) -> None:
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE economic_events (
                id INTEGER PRIMARY KEY, classification TEXT, status TEXT,
                source TEXT, source_url TEXT, evidence TEXT
            );
            CREATE TABLE impact_runs (
                event_id INTEGER, model_id TEXT, prompt_version TEXT, outcome TEXT
            );
            CREATE TABLE impact_errors (
                event_id INTEGER, model_id TEXT, prompt_version TEXT,
                last_error TEXT, attempts INTEGER
            );
            CREATE TABLE market_impacts (
                id INTEGER PRIMARY KEY, event_id INTEGER, model_id TEXT,
                prompt_version TEXT, market TEXT, sector TEXT, direction TEXT,
                horizon TEXT, rationale TEXT, evidence_excerpt TEXT,
                uncertainty TEXT, drivers_json TEXT, entities_json TEXT
            );
            INSERT INTO economic_events VALUES
                (8,'EARNINGS','UNVERIFIED','news','https://example.org/8',
                 'Officially reported improvement'),
                (9,'SUPPLY_CHAIN','UNVERIFIED','news','https://example.org/9',
                 'Reported supplier disruption');
            INSERT INTO impact_runs VALUES
                (8,'gemma2:latest','market-impact-v8','COMPLETED'),
                (9,'gemma2:latest','market-impact-v8','NO_HYPOTHESIS');
            INSERT INTO market_impacts VALUES
                (1,8,'gemma2:latest','market-impact-v8','GLOBAL','FINANCIALS',
                 'UNCERTAIN','MEDIUM_TERM','A conditional market mechanism.',
                 'Officially reported improvement','HIGH','[]','[]');
        """)


def test_exact_model_and_version_source_review_is_read_only(tmp_path: Path) -> None:
    path = tmp_path / "events.sqlite"
    fixture(path)
    before = path.read_bytes()
    report = cohort_review(path, event_ids=(8, 9), model_id="gemma2:latest")
    assert report["status"] == "ok"
    assert report["quality_verified"] is False
    assert report["selected_event_ids"] == [8, 9]
    assert report["items"][0]["outcome"] == "COMPLETED"
    assert report["items"][0]["hypotheses"][0]["direction"] == "UNCERTAIN"
    assert report["items"][0]["hypotheses"][0]["drivers"] == []
    assert report["items"][1]["outcome"] == "NO_HYPOTHESIS"
    assert report["items"][1]["hypotheses"] == []
    assert report["items"][0]["human_label"] == ""
    assert path.read_bytes() == before
    other = cohort_review(path, event_ids=(8,), model_id="other-model")
    assert other["items"][0]["outcome"] == "NOT_ATTEMPTED"
    assert other["items"][0]["hypotheses"] == []


def test_missing_db_and_invalid_ids_do_not_create_files(tmp_path: Path) -> None:
    path = tmp_path / "absent.sqlite"
    assert cohort_review(path, event_ids=(8,), model_id="gemma2:latest")[
        "status"
    ] == "database_missing"
    assert not path.exists()
    with pytest.raises(ValueError):
        cohort_review(path, event_ids=(8, 8), model_id="gemma2:latest")


def test_private_review_created_once_without_leaking_permissions(tmp_path: Path) -> None:
    db = tmp_path / "events.sqlite"
    fixture(db)
    data = cohort_review(db, event_ids=(8,), model_id="gemma2:latest")
    folder = tmp_path / "reports"
    path = save_cohort_review(folder, "cohort-review-sample.json", data)
    assert json.loads(path.read_text())["items"][0]["original_evidence"] == (
        "Officially reported improvement"
    )
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        save_cohort_review(folder, "cohort-review-sample.json", data)
    with pytest.raises(ValueError):
        save_cohort_review(folder, "../cohort-review-outside.json", data)
