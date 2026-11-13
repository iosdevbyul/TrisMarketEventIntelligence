"""Phase 6-F: runtime audit never invokes start/stop or changes SQLite."""

import json
import sqlite3
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from tris_market_event_intelligence.runtime_audit import _api_probe, _launchctl, audit

NOW = datetime.now(UTC)


def _setup(tmp_path: Path) -> tuple[Path, Path]:
    db = tmp_path / "news.sqlite3"
    with sqlite3.connect(db) as connection:
        connection.execute("CREATE TABLE sample(id INTEGER)")
    status = tmp_path / "last-run.json"
    status.write_text(json.dumps({"status": "healthy", "finished_at": NOW.isoformat()}))
    (tmp_path / "run-history.jsonl").write_text(
        json.dumps({"status": "healthy", "finished_at": "2026-11-13T20:12:00+00:00"})
        + "\n" + json.dumps({
            "status": "healthy", "finished_at": "2026-11-13T20:42:00+00:00",
        }) + "\n"
    )
    return db, status


def _launch(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["launchctl"], returncode, stdout, "")


def test_collector_one_shot_not_running_is_normal() -> None:
    output = "state = not running\nruns = 2\nlast exit code = 0\n    state = active\n"
    with patch(
        "tris_market_event_intelligence.runtime_audit.subprocess.run",
        return_value=_launch(output),
    ) as run:
        result = _launchctl("com.tris.market-event-intelligence", 501)
    assert result["state"] == "not running"
    assert result["runs"] == 2
    assert result["last_exit_code"] == "0"
    assert run.call_args.args[0] == [
        "launchctl", "print", "gui/501/com.tris.market-event-intelligence",
    ]


def test_audit_reports_two_cycles_without_changing_state(tmp_path: Path) -> None:
    db, status = _setup(tmp_path)
    previous = (db.read_bytes(), status.read_bytes())
    with (
        patch("tris_market_event_intelligence.runtime_audit._launchctl") as launch,
        patch("tris_market_event_intelligence.runtime_audit._api_probe") as probe,
    ):
        launch.side_effect = [
            {"level": "ok", "reason": "agent_loaded", "state": "not running",
             "runs": 2, "last_exit_code": "0"},
            {"level": "ok", "reason": "agent_loaded", "state": "running",
             "runs": 1, "last_exit_code": "(never exited)"},
        ]
        probe.return_value = {"level": "ok", "reason": "claims_ready"}
        result = audit(db, status, user_id=501)
    assert result["status"] == "ok"
    assert result["checks"]["recurrence"]["reason"] == "multiple_completed_cycles"
    assert result["checks"]["collector_agent"]["level"] == "ok"
    assert previous == (db.read_bytes(), status.read_bytes())
    assert result["modified_launchd"] is False


def test_audit_does_not_claim_recurrence_from_one_history_row(tmp_path: Path) -> None:
    db, status = _setup(tmp_path)
    (tmp_path / "run-history.jsonl").write_text(
        json.dumps({"status": "healthy", "finished_at": NOW.isoformat()}) + "\n"
    )
    with (
        patch("tris_market_event_intelligence.runtime_audit._launchctl") as launch,
        patch("tris_market_event_intelligence.runtime_audit._api_probe") as probe,
    ):
        launch.return_value = {
            "level": "ok", "reason": "agent_loaded", "state": "running",
            "runs": 2, "last_exit_code": "0",
        }
        probe.return_value = {"level": "ok", "reason": "claims_ready"}
        result = audit(db, status, user_id=501)
    assert result["status"] == "warning"
    assert result["checks"]["recurrence"]["reason"] == "insufficient_history"


def test_api_agent_not_running_is_failure(tmp_path: Path) -> None:
    db, status = _setup(tmp_path)
    with (
        patch("tris_market_event_intelligence.runtime_audit._launchctl") as launch,
        patch("tris_market_event_intelligence.runtime_audit._api_probe") as probe,
    ):
        launch.return_value = {
            "level": "ok", "reason": "agent_loaded", "state": "not running",
            "runs": 2, "last_exit_code": "0",
        }
        probe.return_value = {"level": "ok", "reason": "claims_ready"}
        result = audit(db, status, user_id=501)
    assert result["status"] == "error"
    assert result["checks"]["api_agent"]["reason"] == "api_agent_not_running"


def test_invalid_inputs_are_rejected(tmp_path: Path) -> None:
    db, status = _setup(tmp_path)
    with pytest.raises(ValueError, match="Invalid audit"):
        audit(db, status, user_id=501, port=0)
    with pytest.raises(ValueError, match="Invalid audit"):
        audit(db, status, user_id=501, minimum_runs=1)


def test_probe_refuses_unavailable_api() -> None:
    with patch(
        "tris_market_event_intelligence.runtime_audit.urllib.request.urlopen",
        side_effect=OSError("unavailable"),
    ):
        assert _api_probe(8100)["reason"] == "api_unreachable_or_not_ready"
