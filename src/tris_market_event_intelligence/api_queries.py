"""Read-only incremental event and impact queries for local consumers."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path


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
        with closing(self._connection()) as db:
            # All values bound; IDs are stable per database instance.
            rows = db.execute(
                """
                SELECT e.id, e.article_id, e.extractor_version, e.classification, e.status,
                       e.direction, e.change_basis_points, e.evidence, e.source, e.source_url,
                       e.published_at, e.detected_at, e.processed_at, e.capture_sha256,
                       m.claim_kind, m.tags_json, m.verification_note
                FROM economic_events e
                LEFT JOIN event_metadata m ON m.event_id = e.id
                WHERE e.id > ? AND (? IS NULL OR e.status = ?)
                ORDER BY e.id ASC LIMIT ?
                """,
                (after_id, status, status, limit + 1),
            ).fetchall()
        has_more = len(rows) > limit
        items = []
        for row in rows[:limit]:
            item = dict(row)
            item["claimKind"] = item.pop("claim_kind")
            tags = item.pop("tags_json")
            item["tags"] = json.loads(tags) if tags else []
            item["verificationNote"] = item.pop("verification_note")
            items.append(item)
        return {
            "schemaVersion": "1.0",
            "items": items,
            "nextCursor": items[-1]["id"] if items else after_id,
            "hasMore": has_more,
        }

    def claims(self, *, after_id: int, limit: int) -> dict[str, object]:
        with closing(self._connection()) as db:
            rows = db.execute(
                """
                SELECT id, article_id, extractor_version, ordinal, classification,
                       tag, claim_kind, polarity, temporal_context, status, evidence,
                       verification_note, source, source_url, published_at,
                       detected_at, processed_at, capture_sha256
                FROM article_event_claims WHERE id > ?
                ORDER BY id ASC LIMIT ?
                """,
                (after_id, limit + 1),
            ).fetchall()
        has_more = len(rows) > limit
        items = [dict(row) for row in rows[:limit]]
        return {
            "schemaVersion": "1.0", "items": items,
            "nextCursor": items[-1]["id"] if items else after_id,
            "hasMore": has_more,
        }

    def latest_claim_reviews(self, *, after_id: int, limit: int) -> dict[str, object]:
        # Latest review per claim, with claim-ID cursor (not history revision ID).
        with closing(self._connection()) as db:
            rows = db.execute(
                """
                SELECT r.*, c.classification, c.evidence AS original_evidence
                FROM claim_review_history r
                JOIN article_event_claims c ON c.id = r.claim_id
                WHERE r.claim_id > ?
                  AND r.revision = (
                    SELECT MAX(r2.revision) FROM claim_review_history r2
                    WHERE r2.claim_id = r.claim_id
                  )
                ORDER BY r.claim_id LIMIT ?
                """,
                (after_id, limit + 1),
            ).fetchall()
        has_more = len(rows) > limit
        items = [dict(row) for row in rows[:limit]]
        return {
            "schemaVersion": "1.0", "items": items,
            "nextCursor": items[-1]["claim_id"] if items else after_id,
            "hasMore": has_more,
        }

    def claim_review_history(
        self, *, after_id: int, limit: int, claim_id: int | None
    ) -> dict[str, object]:
        with closing(self._connection()) as db:
            rows = db.execute(
                """
                SELECT id, operation_id, claim_id, revision, prior_status, status,
                       evidence_article_id, evidence_quote, evidence_kind,
                       reviewer, reason, created_at
                FROM claim_review_history
                WHERE id > ? AND (? IS NULL OR claim_id = ?)
                ORDER BY id LIMIT ?
                """,
                (after_id, claim_id, claim_id, limit + 1),
            ).fetchall()
        has_more = len(rows) > limit
        items = [dict(row) for row in rows[:limit]]
        return {
            "schemaVersion": "1.0", "items": items,
            "nextCursor": items[-1]["id"] if items else after_id,
            "hasMore": has_more,
        }

    def analyses(self, *, after_id: int, limit: int) -> dict[str, object]:
        with closing(self._connection()) as db:
            rows = db.execute(
                """
                SELECT i.id, i.event_id, i.model_id, i.prompt_version, i.market,
                       i.sector, i.direction, i.horizon, i.rationale,
                       i.evidence_excerpt, i.uncertainty, i.drivers_json,
                       i.generated_at, i.entities_json, e.status AS event_status,
                       m.claim_kind
                FROM market_impacts i
                JOIN economic_events e ON e.id = i.event_id
                LEFT JOIN event_metadata m ON m.event_id = e.id
                WHERE i.id > ?
                ORDER BY i.id ASC LIMIT ?
                """,
                (after_id, limit + 1),
            ).fetchall()
        has_more = len(rows) > limit
        items = []
        for row in rows[:limit]:
            entry = dict(row)
            entry["drivers"] = json.loads(entry.pop("drivers_json"))
            entry["entities"] = json.loads(entry.pop("entities_json"))
            entry["eventStatus"] = entry.pop("event_status")
            entry["claimKind"] = entry.pop("claim_kind")
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
        with closing(self._connection()) as db:
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
        with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)) as db:
            required = {"economic_events", "market_impacts", "impact_runs"}
            found = {row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
            return required.issubset(found)
    except sqlite3.Error:
        return False


def claims_database_ready(db_path: Path) -> bool:
    """True only if the collector's claim tables exist; no schema creation."""
    if not db_path.is_file():
        return False
    try:
        with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)) as db:
            db.execute("PRAGMA query_only=ON")
            required = {"articles", "captures", "article_event_claims"}
            found = {
                row[0] for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            return required.issubset(found)
    except sqlite3.Error:
        return False
