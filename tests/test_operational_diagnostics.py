"""Offline health and unexpected collection failure regression tests."""
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from tris_market_event_intelligence.operational_diagnostics import diagnose
from tris_market_event_intelligence.operations import run_cycle

NOW = datetime(2026, 11, 14, 0, 0, tzinfo=UTC)


def _fixture(tmp_path: Path, *, status: str = "healthy", age: int = 10) -> tuple[Path, Path]:
    db_path = tmp_path / "news.sqlite3"
    with sqlite3.connect(db_path) as db:
        db.execute("CREATE TABLE sample (id INTEGER)")
    path = tmp_path / "last-run.json"
    path.write_text(json.dumps({
        "status": status,
        "finished_at": (NOW - timedelta(minutes=age)).isoformat(),
    }))
    return db_path, path


def test_diagnostic_reports_healthy_and_never_mutates_db(tmp_path: Path) -> None:
    db, status = _fixture(tmp_path)
    before = db.read_bytes()
    report = diagnose(db, status, now=NOW)
    assert report["status"] == "ok"
    assert report["checks"]["database"]["reason"] == "database_valid"
    assert report["network_requests"] == 0
    assert db.read_bytes() == before


def test_diagnostic_flags_stale_failed_and_degraded(tmp_path: Path) -> None:
    db, path = _fixture(tmp_path, age=100)
    assert diagnose(db, path, now=NOW)["checks"]["last_run"]["reason"] == "stale_run"
    db, path = _fixture(tmp_path, status="failed")
    assert diagnose(db, path, now=NOW)["status"] == "error"
    db, path = _fixture(tmp_path, status="degraded")
    assert diagnose(db, path, now=NOW)["status"] == "warning"


def test_diagnostic_rejects_bad_status_and_corrupt_database(tmp_path: Path) -> None:
    db, status = _fixture(tmp_path)
    status.write_text("{broken")
    assert diagnose(db, status, now=NOW)["checks"]["last_run"]["reason"] == "invalid_status_json"
    db.write_bytes(b"not a sqlite database")
    assert diagnose(db, status, now=NOW)["checks"]["database"]["level"] == "error"


def test_diagnostic_reports_storage_threshold(tmp_path: Path) -> None:
    db, status = _fixture(tmp_path)
    result = diagnose(db, status, now=NOW, warning_bytes=1)
    assert result["checks"]["database"]["reason"] == "database_large"
    assert result["status"] == "warning"


def test_exception_overwrites_previous_healthy_checkpoint(tmp_path: Path) -> None:
    db, status = _fixture(tmp_path)
    with patch(
        "tris_market_event_intelligence.operations.NewsStore.initialize",
        side_effect=RuntimeError("SECRET_TOKEN_NOT_FOR_LOGS"),
    ):
        result = run_cycle(db, status_path=status, selected_sources=("fed_all",))
    assert result["status"] == "failed"
    assert result["error_code"] == "RuntimeError"
    contents = status.read_text()
    assert "SECRET_TOKEN_NOT_FOR_LOGS" not in contents
    assert json.loads(contents)["status"] == "failed"
