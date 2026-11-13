"""Read-only access to prior impact model failures; never retries inference."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path


def inspect_errors(db_path: Path, *, limit: int = 10) -> dict[str, object]:
    """Only expose saved, sanitized exception class names and retry counts.

    No raw provider errors, prompts, evidence or source URLs are returned.
    The SQLite URI mode=ro guarantees a nonexistent database isn't created.
    """
    if not 1 <= limit <= 100:
        raise ValueError("limit must be 1..100")
    if not db_path.is_file():
        return {"status": "database_missing", "errors": []}
    try:
        with closing(sqlite3.connect(
            f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True, timeout=5,
        )) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only=ON")
            table_exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='impact_errors'"
            ).fetchone()
            if table_exists is None:
                return {"status": "not_initialized", "errors": []}
            rows = conn.execute(
                """SELECT event_id, model_id, prompt_version,
                          last_error, last_attempt_at, attempts
                   FROM impact_errors
                   ORDER BY last_attempt_at DESC, event_id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
    except sqlite3.Error:
        return {"status": "database_unreadable", "errors": []}
    return {"status": "ok", "errors": [dict(row) for row in rows]}
