"""Read-only incremental event and impact queries for local consumers."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from tris_market_event_intelligence.event_store import EventProcessor


class EventReader:
    """Independent monotonically increasing cursors avoid late-analysis data loss."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def _connection(self) -> sqlite3.Connection:
        if not self.db_path.is_file():
            raise FileNotFoundError("Event database is not initialized")
        db = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        return db

    def events(
        self, *, after_id: int, limit: int, status: str | None
    ) -> dict[str, object]:
        with self._connection() as db:
            # All values bound; IDs are stable per database instance.
            rows = db.execute(
                """
                SELECT id, article_id, extractor_version, classification, status,
                       direction, change_basis_points, evidence, source, source_url,
                       published_at, detected_at, processed_at, capture_sha256
                FROM economic_events
                WHERE id > ? AND (? IS NULL OR status = ?)
                ORDER BY id ASC LIMIT ?
                """,
                (after_id, status, status, limit + 1),
            ).fetchall()
        has_more = len(rows) > limit
        items = [dict(r) for r in rows[:limit]]
        return {
            "schemaVersion": "1.0",
            "items": items,
            "nextCursor": items[-1]["id"] if items else after_id,
            "hasMore": has_more,
        }

    def analyses(self, *, after_id: int, limit: int) -> dict[str, object]:
        with self._connection() as db:
            rows = db.execute(
                """
                SELECT i.id, i.event_id, i.model_id, i.prompt_version, i.market,
                       i.sector, i.direction, i.horizon, i.rationale,
                       i.evidence_excerpt, i.uncertainty, i.drivers_json,
                       i.generated_at
                FROM market_impacts i
                ORDER BY i.id ASC
                LIMIT ?
                """,
                (limit + 1,),
            ).fetchall() if after_id == 0 else db.execute(
                """
                SELECT id, event_id, model_id, prompt_version, market,
                       sector, direction, horizon, rationale, evidence_excerpt,
                       uncertainty, drivers_json, generated_at
                FROM market_impacts
                WHERE id > ?
                ORDER BY id ASC LIMIT ?
                """,
                (after_id, limit + 1),
            ).fetchall()
        has_more = len(rows) > limit
        items = []
        for row in rows[:limit]:
            entry = dict(row)
            entry["drivers"] = json.loads(entry.pop("drivers_json"))
            items.append(entry)
        return {
            "schemaVersion": "1.0",
            "items": items,
            "nextCursor": items[-1]["id"] if items else after_id,
            "hasMore": has_more,
        }

    def analysis_runs(
        self, *, after_rowid: int, limit: int
    ) -> dict[str, object]:
        # Even NO_HYPOTHESIS should be visible to consumers separately.
        with self._connection() as db:
            rows = db.execute(
                """
                SELECT rowid AS id, event_id, model_id, prompt_version,
                       processed_at, outcome
                FROM impact_runs WHERE rowid > ?
                ORDER BY rowid ASC LIMIT ?
                """,
                (after_rowid, limit + 1),
            ).fetchall()
        has_more = len(rows) > limit
        items = [dict(r) for r in rows[:limit]]
        return {
            "schemaVersion": "1.0",
            "items": items,
            "nextCursor": items[-1]["id"] if items else after_rowid,
            "hasMore": has_more,
        }


def database_ready(db_path: Path) -> bool:
    if not db_path.is_file():
        return False
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5) as db:
            required = {"economic_events", "market_impacts", "impact_runs"}
            found = {row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
            return required.issubset(found)
    except sqlite3.Error:
        return False
