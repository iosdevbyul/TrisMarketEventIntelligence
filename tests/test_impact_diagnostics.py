"""Stored model failure inspection must not rerun inference or write SQLite."""

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from tris_market_event_intelligence.impact_cli import main as cli_main
from tris_market_event_intelligence.impact_diagnostics import inspect_errors


def test_missing_db_is_not_created(tmp_path: Path) -> None:
    db = tmp_path / "missing.sqlite3"
    assert inspect_errors(db) == {"status": "database_missing", "errors": []}
    assert not db.exists()


def test_uninitialized_db_is_not_modified(tmp_path: Path) -> None:
    db = tmp_path / "test.sqlite3"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE unrelated(id INTEGER)")
    original = db.read_bytes()
    assert inspect_errors(db) == {"status": "not_initialized", "errors": []}
    assert db.read_bytes() == original


def test_read_only_errors_are_sanitized_and_limited(tmp_path: Path) -> None:
    db = tmp_path / "test.sqlite3"
    with sqlite3.connect(db) as conn:
        conn.execute("""CREATE TABLE impact_errors (
            event_id INTEGER, model_id TEXT, prompt_version TEXT,
            last_error TEXT, last_attempt_at TEXT, attempts INTEGER
        )""")
        conn.executemany(
            "INSERT INTO impact_errors VALUES (?, ?, ?, ?, ?, ?)",
            [
                (1, "gemma2:latest", "market-impact-v2", "ReadTimeout", "a", 1),
                (3, "gemma2:latest", "market-impact-v2", "ValidationError", "b", 2),
            ],
        )
    before = db.read_bytes()
    with patch("sys.argv", [
        "market-event-analyze", "--db", str(db), "--show-errors", "--error-limit", "1",
    ]), patch.dict("os.environ", {"MARKET_EVENT_LLM_MODEL": ""}):
        with patch(
            "tris_market_event_intelligence.impact_cli.OpenAICompatibleProvider",
        ) as model:
            from io import StringIO
            from contextlib import redirect_stdout
            output = StringIO()
            with redirect_stdout(output):
                cli_main()
            model.assert_not_called()
    payload = json.loads(output.getvalue())
    assert payload["status"] == "ok"
    assert payload["errors"] == [{
        "event_id": 3,
        "model_id": "gemma2:latest",
        "prompt_version": "market-impact-v2",
        "last_error": "ValidationError",
        "last_attempt_at": "b",
        "attempts": 2,
    }]
    assert db.read_bytes() == before


@pytest.mark.parametrize("limit", [0, 101])
def test_invalid_error_limit(limit: int, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="limit"):
        inspect_errors(tmp_path / "test.sqlite3", limit=limit)
