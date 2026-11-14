"""Paired v7/v8 review samples preserve source and abstention without inference."""
import json
import os
import sqlite3
import stat
from pathlib import Path

import pytest

from tris_market_event_intelligence.impact_pair_review import (
    paired_review_samples,
    save_private_review_bundle,
)


def make_db(path: Path) -> None:
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE economic_events (
                id INTEGER PRIMARY KEY, classification TEXT, status TEXT,
                source TEXT, source_url TEXT, evidence TEXT
            );
            CREATE TABLE impact_runs (
                event_id INTEGER, model_id TEXT, prompt_version TEXT, outcome TEXT
            );
            CREATE TABLE market_impacts (
                id INTEGER PRIMARY KEY, event_id INTEGER, model_id TEXT,
                prompt_version TEXT, market TEXT, sector TEXT, direction TEXT,
                horizon TEXT, rationale TEXT, evidence_excerpt TEXT,
                uncertainty TEXT
            );
            INSERT INTO economic_events VALUES
                (13, 'EARNINGS', 'UNVERIFIED', 'news',
                 'https://example.org/13',
                 '□ 2026년 2/4분기 외감기업의 성장성, 수익성, 안정성이 모두 개선'),
                (7, 'ECONOMIC_INDICATOR', 'UNVERIFIED', 'news',
                 'https://example.org/7', 'Inflation changed'),
                (18, 'SUPPLY_CHAIN', 'UNVERIFIED', 'news',
                 'https://example.org/18', 'Supply chain claim');
            INSERT INTO impact_runs VALUES
                (13,'gemma2:latest','market-impact-v7','COMPLETED'),
                (13,'gemma2:latest','market-impact-v8','NO_HYPOTHESIS'),
                (7,'gemma2:latest','market-impact-v7','NO_HYPOTHESIS'),
                (7,'gemma2:latest','market-impact-v8','NO_HYPOTHESIS'),
                (18,'gemma2:latest','market-impact-v7','COMPLETED'),
                (18,'other-model','market-impact-v8','COMPLETED');
            INSERT INTO market_impacts VALUES
                (1,13,'gemma2:latest','market-impact-v7','GLOBAL','BROAD_MARKET',
                 'UNCERTAIN','MEDIUM_TERM',
                 'Improved financial performance may indicate broader economic strength.',
                 '성장성, 수익성, 안정성이 모두 개선','HIGH'),
                (2,18,'gemma2:latest','market-impact-v7','KR','TECH',
                 'UNCERTAIN','SHORT_TERM','Unverified story',
                 'Supply chain claim','HIGH');
        """)


def test_pairs_are_source_linked_and_version_isolated_and_read_only(tmp_path: Path) -> None:
    db = tmp_path / "events.sqlite"
    make_db(db)
    before = db.read_bytes()
    result = paired_review_samples(db, model_id="gemma2:latest")
    assert result["status"] == "ok"
    assert result["quality_verified"] is False
    assert result["sampled_events"] == 2
    assert [row["event_id"] for row in result["items"]] == [13, 7]
    first = result["items"][0]
    assert first["baseline_outcome"] == "COMPLETED"
    assert first["candidate_outcome"] == "NO_HYPOTHESIS"
    assert "외감기업" in first["original_evidence"]
    assert len(first["baseline_hypotheses"]) == 1
    assert first["candidate_hypotheses"] == []
    assert first["human_label"] == ""
    assert first["human_review_notes"] == ""
    assert db.read_bytes() == before


def test_private_local_output_and_no_overwrite(tmp_path: Path) -> None:
    db = tmp_path / "events.sqlite"
    make_db(db)
    result = paired_review_samples(db, model_id="gemma2:latest")
    folder = tmp_path / "review"
    path = save_private_review_bundle(folder, "paired-review-test.json", result)
    assert "외감기업" in path.read_text()
    assert json.loads(path.read_text())["items"][0]["event_id"] == 13
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        save_private_review_bundle(folder, "paired-review-test.json", result)
    with pytest.raises(ValueError):
        save_private_review_bundle(folder, "../paired-review-leak.json", result)


def test_unpaired_versions_not_mistaken_for_valid_abstention(tmp_path: Path) -> None:
    db = tmp_path / "events.sqlite"
    make_db(db)
    result = paired_review_samples(db, model_id="other-model")
    assert result["sampled_events"] == 0
    assert result["items"] == []


def test_missing_db_and_invalid_versions(tmp_path: Path) -> None:
    db = tmp_path / "missing.sqlite3"
    result = paired_review_samples(db, model_id="gemma2:latest")
    assert result["status"] == "database_missing"
    assert not db.exists()
    with pytest.raises(ValueError):
        paired_review_samples(db, model_id="model", baseline="v8", candidate="v8")
    with pytest.raises(ValueError):
        paired_review_samples(db, model_id="model", limit=6)
