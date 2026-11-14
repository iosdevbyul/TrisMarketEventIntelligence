"""Durable, retryable, append-only market-impact analysis."""

from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from tris_market_event_intelligence.collector import utc_now
from tris_market_event_intelligence.event_store import EventProcessor
from tris_market_event_intelligence.impact import (
    PROMPT_VERSION,
    EventInput,
    ImpactAnalysisError,
    ImpactProvider,
    is_eligible,
    validate_result,
)
from tris_market_event_intelligence.source_health import assess_live_event, audit_source_health


@dataclass(frozen=True)
class AnalysisStats:
    processed: int
    impacts: int
    failed: int
    deferred: int = 0
    already_running: bool = False


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
                    entities_json TEXT NOT NULL DEFAULT '[]',
                    UNIQUE(event_id, model_id, prompt_version, market, sector, horizon)
                );
                CREATE TABLE IF NOT EXISTS impact_attempt_timings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER NOT NULL REFERENCES economic_events(id),
                    model_id TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    attempted_at TEXT NOT NULL,
                    provider_returned_at TEXT,
                    completed_at TEXT NOT NULL,
                    provider_elapsed_ms REAL,
                    processing_elapsed_ms REAL NOT NULL,
                    outcome TEXT NOT NULL CHECK(outcome IN ('COMPLETED','NO_HYPOTHESIS','ERROR'))
                );
                CREATE INDEX IF NOT EXISTS idx_impact_attempt_timings_event
                  ON impact_attempt_timings(event_id, attempted_at);
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
            if "entities_json" not in {
                row[1] for row in db.execute("PRAGMA table_info(market_impacts)")
            }:
                db.execute(
                    "ALTER TABLE market_impacts ADD COLUMN entities_json "
                    "TEXT NOT NULL DEFAULT '[]'"
                )

    def process_pending(
        self, batch_size: int = 100, *, event_ids: tuple[int, ...] | None = None,
        live_freshness_seconds: int | None = None,
        source_success_seconds: int = 900,
    ) -> AnalysisStats:
        """Hold a nonblocking per-DB lock across selection, model calls and checkpoint.

        This prevents separate CLI/process invocations from invoking the same
        uncheckpointed event concurrently, without touching main launchd.
        """
        if not 1 <= batch_size <= 1000:
            raise ValueError("batch_size must be 1..1000")
        if event_ids is not None and (
            not 1 <= len(event_ids) <= 3
            or len(set(event_ids)) != len(event_ids)
            or any(type(i) is not int or i < 1 for i in event_ids)
            or batch_size < len(event_ids)
        ):
            raise ValueError("Select 1..3 unique positive event IDs within batch size")
        if live_freshness_seconds is not None and (
            type(live_freshness_seconds) is not int or live_freshness_seconds < 1
        ):
            raise ValueError("live_freshness_seconds must be positive")
        if type(source_success_seconds) is not int or source_success_seconds < 1:
            raise ValueError("source_success_seconds must be positive")
        if not self.provider.model_id.strip():
            raise ValueError("Provider requires a stable model_id")
        path = self.store.store.path
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_name(path.name + ".analysis.lock")
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        with os.fdopen(fd, "w") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return AnalysisStats(
                    processed=0, impacts=0, failed=0, already_running=True,
                )
            return self._process_pending_unlocked(
                batch_size, event_ids=event_ids,
                live_freshness_seconds=live_freshness_seconds,
                source_success_seconds=source_success_seconds,
            )

    def _process_pending_unlocked(
        self, batch_size: int = 100, *, event_ids: tuple[int, ...] | None = None,
        live_freshness_seconds: int | None = None,
        source_success_seconds: int = 900,
    ) -> AnalysisStats:
        if not 1 <= batch_size <= 1000:
            raise ValueError("batch_size must be 1..1000")
        if not self.provider.model_id.strip():
            raise ValueError("Provider requires a stable model_id")
        if event_ids is not None and (
            not 1 <= len(event_ids) <= 3
            or len(set(event_ids)) != len(event_ids)
            or any(type(event_id) is not int or event_id < 1 for event_id in event_ids)
            or batch_size < len(event_ids)
        ):
            raise ValueError("Select 1..3 unique positive event IDs within batch size")
        if live_freshness_seconds is not None and (
            type(live_freshness_seconds) is not int or live_freshness_seconds < 1
        ):
            raise ValueError("live_freshness_seconds must be positive")
        if type(source_success_seconds) is not int or source_success_seconds < 1:
            raise ValueError("source_success_seconds must be positive")
        self.initialize()
        selection_sql = (
            " AND e.id IN (" + ",".join("?" for _ in event_ids) + ")"
            if event_ids is not None else ""
        )
        with self.store.store.connect() as db:
            rows = db.execute(
                f"""
                SELECT e.id, e.classification, e.status, e.direction,
                       e.change_basis_points, e.evidence, e.source, e.source_url,
                       e.detected_at, m.claim_kind, m.verification_note, m.tags_json,
                       e.published_at
                FROM economic_events e
                LEFT JOIN event_metadata m ON m.event_id = e.id
                WHERE e.classification IN (
                    'MONETARY_POLICY', 'ECONOMIC_INDICATOR', 'EARNINGS',
                    'CORPORATE_EVENT', 'REGULATION', 'GEOPOLITICAL', 'SUPPLY_CHAIN',
                    'INDUSTRY', 'SERVICE_INCIDENT', 'MARKET_MOVEMENT'
                )
                  AND (
                    (e.status='VERIFIED' AND e.classification='MONETARY_POLICY')
                    OR (
                        e.status IN ('CANDIDATE','UNVERIFIED')
                        AND e.classification!='MONETARY_POLICY'
                        AND m.claim_kind='REPORTED'
                    )
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM impact_runs r
                    WHERE r.event_id=e.id AND r.model_id=? AND r.prompt_version=?
                  )
                {selection_sql}
                ORDER BY e.id LIMIT ?
                """,
                (self.provider.model_id, PROMPT_VERSION, *(event_ids or ()), batch_size),
            ).fetchall()
        if event_ids is not None and {row[0] for row in rows} != set(event_ids):
            raise ValueError("Selected event is missing, ineligible, or already checkpointed")

        processed = impacts = failed = deferred = 0
        health_by_source: dict[str, str] = {}
        live_now = datetime.now(UTC)
        if live_freshness_seconds is not None and rows:
            health = audit_source_health(
                self.store.store.path,
                expected_sources=tuple(sorted({row[6] for row in rows})),
                now=live_now,
                max_success_age_seconds=source_success_seconds,
            )
            health_by_source = {
                item["source"]: item["status"] for item in health["sources"]
            }
        for record in rows:
            event = EventInput(
                *record[:9], record[9], record[10],
                tuple(json.loads(record[11])) if record[11] else (),
            )
            if not is_eligible(event):
                continue
            if live_freshness_seconds is not None:
                # Every model request gets its OWN current-clock check.
                # An earlier slow Ollama call must not cause a later, now-stale
                # event to be sent based on a frozen batch-start timestamp.
                current = datetime.now(UTC)
                decision = assess_live_event(
                    detected_at=event.detected_at,
                    published_at=record[12],
                    now=current,
                    max_received_age_seconds=live_freshness_seconds,
                    max_declared_publish_age_seconds=live_freshness_seconds,
                )
                if decision != "ELIGIBLE_WITH_UNVERIFIED_PUBLICATION" or (
                    health_by_source.get(event.source) != "OK"
                ):
                    # Deferred: never call the model or checkpoint a stale event.
                    deferred += 1
                    continue
            started_at = utc_now()
            started_mono = time.monotonic()
            provider_returned_at: str | None = None
            provider_elapsed_ms: float | None = None
            outcome = "ERROR"
            try:
                try:
                    raw_result = self.provider.analyze(event)
                finally:
                    # Capture elapsed provider time even if Ollama times out or
                    # the provider raises before returning an answer.
                    provider_returned_at = utc_now()
                    provider_elapsed_ms = max(
                        0.0, (time.monotonic() - started_mono) * 1000,
                    )
                result = validate_result(event, raw_result)
                keys = [(i.market, i.sector, i.horizon) for i in result.impacts]
                if len(keys) != len(set(keys)):
                    raise ImpactAnalysisError(
                        "DUPLICATE_IMPACT_KEYS", "Duplicate market/sector/horizon impact keys"
                    )
                now = utc_now()
                with self.store.store.connect() as db:
                    for impact in result.impacts:
                        db.execute(
                            """
                            INSERT INTO market_impacts (
                                event_id, model_id, prompt_version, market, sector,
                                direction, horizon, rationale, evidence_excerpt,
                                uncertainty, drivers_json, generated_at, entities_json
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (event.event_id, self.provider.model_id, PROMPT_VERSION,
                             impact.market, impact.sector, impact.direction, impact.horizon,
                             impact.rationale, impact.evidence_excerpt, impact.uncertainty,
                             json.dumps(impact.drivers, ensure_ascii=False), now,
                             json.dumps(impact.entities, ensure_ascii=False)),
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
                outcome = "COMPLETED" if result.impacts else "NO_HYPOTHESIS"
            except (ValueError, sqlite3.Error, KeyError, IndexError, TypeError) as exc:
                self._record_error(event.event_id, exc)
                failed += 1
            except Exception as exc:  # noqa: BLE001
                # Isolate arbitrary pluggable provider failures; never checkpoint them.
                self._record_error(event.event_id, exc)
                failed += 1
            finally:
                # Append only, preserving precise duration even if a model
                # produced invalid output or database checkpointing failed.
                completed_at = utc_now()
                with self.store.store.connect() as db:
                    db.execute(
                        """INSERT INTO impact_attempt_timings (
                            event_id,model_id,prompt_version,attempted_at,
                            provider_returned_at,completed_at,provider_elapsed_ms,
                            processing_elapsed_ms,outcome
                        ) VALUES (?,?,?,?,?,?,?,?,?)""",
                        (
                            event.event_id, self.provider.model_id, PROMPT_VERSION,
                            started_at, provider_returned_at, completed_at,
                            provider_elapsed_ms,
                            max(0.0, (time.monotonic() - started_mono) * 1000),
                            outcome,
                        ),
                    )
        return AnalysisStats(processed=processed, impacts=impacts, failed=failed, deferred=deferred)

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
                 error.code if isinstance(error, ImpactAnalysisError)
                 else (
                     f"HTTP_{error.response.status_code}"
                     if isinstance(error, httpx.HTTPStatusError)
                     else type(error).__name__
                 ), utc_now()),
            )

    def list_impacts(self) -> list[dict[str, object]]:
        self.initialize()
        with self.store.store.connect() as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("SELECT * FROM market_impacts ORDER BY id").fetchall()
        return [dict(row) for row in rows]
