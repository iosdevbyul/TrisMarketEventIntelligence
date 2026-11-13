"""Failure streaks and recovery are assessed without running collectors."""

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tris_market_event_intelligence.operational_diagnostics import diagnose
from tris_market_event_intelligence.operational_recovery import assess_recovery, recent_runs


NOW = datetime(2026, 11, 14, tzinfo=UTC)


def _history(tmp_path: Path, statuses: list[str]) -> Path:
    file = tmp_path / "run-history.jsonl"
    file.write_text("".join(
        json.dumps({"status": status, "finished_at": str(i)}) + "\n"
        for i, status in enumerate(statuses)
    ))
    return file


def test_failure_streak_escalates_at_threshold(tmp_path: Path) -> None:
    path = _history(tmp_path, ["healthy", "failed", "failed", "failed"])
    result = assess_recovery(path)
    assert result["level"] == "error"
    assert result["reason"] == "repeated_failure"
    assert result["consecutive_failures"] == 3
    assert result["recovered"] is False


def test_success_after_outage_clears_streak_without_erasing_history(tmp_path: Path) -> None:
    path = _history(tmp_path, ["healthy", "failed", "failed", "healthy"])
    before = path.read_bytes()
    result = assess_recovery(path)
    assert result["level"] == "ok"
    assert result["reason"] == "recovered"
    assert result["consecutive_failures"] == 0
    assert result["recovered"] is True
    assert path.read_bytes() == before


def test_degraded_and_recent_failure_are_not_healthy(tmp_path: Path) -> None:
    path = _history(tmp_path, ["healthy", "degraded", "degraded"])
    result = assess_recovery(path)
    assert result["level"] == "warning"
    assert result["consecutive_degraded"] == 2
    path = _history(tmp_path, ["healthy", "failed"])
    result = assess_recovery(path)
    assert result["reason"] == "recent_failure"


def test_missing_and_partial_history_do_not_invent_success(tmp_path: Path) -> None:
    path = tmp_path / "run-history.jsonl"
    assert assess_recovery(path)["reason"] == "history_not_recorded_yet"
    path.write_bytes(b'{"status":"healthy"}\n{"status":"failed"')
    assert recent_runs(path)[0]["status"] == "healthy"
    path.write_bytes(b'{"status":"failed"')
    assert assess_recovery(path)["reason"] == "history_incomplete"


def test_doctor_integrates_failure_streak_without_modifying_database(tmp_path: Path) -> None:
    db = tmp_path / "news.sqlite3"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE sample (id INTEGER)")
    status = tmp_path / "last-run.json"
    status.write_text(json.dumps({
        "status": "failed",
        "finished_at": (NOW - timedelta(minutes=5)).isoformat(),
    }))
    path = _history(tmp_path, ["healthy", "failed", "failed", "failed"])
    original_history = path.read_bytes()
    original_db = db.read_bytes()
    outcome = diagnose(db, status, now=NOW)
    assert outcome["status"] == "error"
    assert outcome["checks"]["recovery"]["consecutive_failures"] == 3
    assert original_history == path.read_bytes()
    assert original_db == db.read_bytes()
