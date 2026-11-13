"""Durable, retryable, append-only market-impact analysis."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from tris_market_event_intelligence.collector import utc_now
from tris_market_event_intelligence.event_store import EventProcessor
from tris_market_event_intelligence.impact import (
    PROMPT_VERSION,
    EventInput,
    ImpactProvider,
    validate_result,
)


@dataclass(frozen=True)
class AnalysisStats:
    processed: int
    impacts: int
    failed: int


class ImpactProcessor:
    def __init__(self, db_path: Path, provider: ImpactProvider) -> None:
        self.store = EventProcessor(db_path)
        self.provider = provider

    def initialize(self) -> None:
        self.store.initialize()
        with self.store.store.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS impact_runs (
                    event_id INTEGER NOT NULL REFERENCES economic_events(id),
                    model_id TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    processed_at TEXT NOT NULL,
                    outcome TEXT NOT NULL CHECK(outcome IN ('COMPLETED', 'NO_HYPOTHESIS')),
                    PRIMARY KEY(event_id, model_id, prompt_version)
                );
                CREATE TABLE IF NOT EXISTS market_impacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL REFERENCES economic_events(id),
                    model_id TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    market TEXT NOT NULL,
                    sector TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    horizon TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    evidence_excerpt TEXT NOT NULL,
                    uncertainty TEXT NOT NULL,
                    drivers_json TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    UNIQUE(event_id, model_id, prompt_version, market, sector, horizon)
                );
                CREATE TABLE IF NOT EXISTS impact_errors (
                    event_id INTEGER NOT NULL REFERENCES economic_events(id),
                    model_id TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    last_error TEXT NOT NULL,
                    last_attempt_at TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY(event_id, model_id, prompt_version)
                );
                """
            )

    def process_pending(self, batch_size: int = 100) -> AnalysisStats:
        if not 1 <= batch_size <= 1000:
            raise ValueError("batch_size must be 1..1000")
        if not self.provider.model_id.strip():
            raise ValueError("Provider requires a stable model_id")
        self.initialize()
        with self.store.store.connect() as db:
            rows = db.execute(
                """
                SELECT e.id, e.classification, e.status, e.direction,
                       e.change_basis_points, e.evidence, e.source, e.source_url,
                       e.detected_at
                FROM economic_events e
                WHERE e.status='VERIFIED' AND e.classification='MONETARY_POLICY'
                  AND NOT EXISTS (
                    SELECT 1 FROM impact_runs r
                    WHERE r.event_id=e.id AND r.model_id=? AND r.prompt_version=?
                  )
                ORDER BY e.id LIMIT ?
                """,
                (self.provider.model_id, PROMPT_VERSION, batch_size),
            ).fetchall()

        processed = impacts = failed = 0
        for record in rows:
            event = EventInput(*record)
            try:
                result = validate_result(event, self.provider.analyze(event))
                keys = [(i.market, i.sector, i.horizon) for i in result.impacts]
                if len(keys) != len(set(keys)):
                    raise ValueError("Duplicate market/sector/horizon impact keys")
                now = utc_now()
                with self.store.store.connect() as db:
                    for impact in result.impacts:
                        db.execute(
                            """
                            INSERT INTO market_impacts (
                                event_id, model_id, prompt_version, market, sector,
                                direction, horizon, rationale, evidence_excerpt,
                                uncertainty, drivers_json, generated_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (event.event_id, self.provider.model_id, PROMPT_VERSION,
                             impact.market, impact.sector, impact.direction, impact.horizon,
                             impact.rationale, impact.evidence_excerpt, impact.uncertainty,
                             json.dumps(impact.drivers, ensure_ascii=False), now),
                        )
                    db.execute(
                        "INSERT INTO impact_runs VALUES (?, ?, ?, ?, ?)",
                        (event.event_id, self.provider.model_id, PROMPT_VERSION, now,
                         "COMPLETED" if result.impacts else "NO_HYPOTHESIS"),
                    )
                    db.execute(
                        "DELETE FROM impact_errors WHERE event_id=? AND model_id=? "
                        "AND prompt_version=?",
                        (event.event_id, self.provider.model_id, PROMPT_VERSION),
                    )
                processed += 1
                impacts += len(result.impacts)
            except (ValueError, sqlite3.Error, KeyError, IndexError, TypeError) as exc:
                self._record_error(event.event_id, exc)
                failed += 1
            except Exception as exc:  # noqa: BLE001
                # Isolate arbitrary pluggable provider failures; never checkpoint them.
                self._record_error(event.event_id, exc)
                failed += 1
        return AnalysisStats(processed=processed, impacts=impacts, failed=failed)

    def _record_error(self, event_id: int, error: Exception) -> None:
        with self.store.store.connect() as db:
            db.execute(
                """
                INSERT INTO impact_errors (
                    event_id, model_id, prompt_version, last_error, last_attempt_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(event_id, model_id, prompt_version) DO UPDATE SET
                    last_error=excluded.last_error,
                    last_attempt_at=excluded.last_attempt_at,
                    attempts=impact_errors.attempts+1
                """,
                (event_id, self.provider.model_id, PROMPT_VERSION,
                 type(error).__name__, utc_now()),
            )

    def list_impacts(self) -> list[dict[str, object]]:
        self.initialize()
        with self.store.store.connect() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("SELECT * FROM market_impacts ORDER BY id").fetchall()
        return [dict(row) for row in rows]
