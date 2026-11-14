"""Read-only attribution of extraction queue, provider work and persistence.

Older impact_runs have only checkpoint timestamps and CANNOT establish
historic local Ollama inference duration. New timings are append-only.
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

MAX_ROWS = 50_000


def _time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None:
        raise ValueError("Explicit timestamp zone required")
    return parsed.astimezone(UTC)


def _stat(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median_seconds": None, "p95_seconds": None}
    ordered = sorted(values)
    rank = (95 * len(ordered) + 99) // 100 - 1
    return {
        "count": len(ordered),
        "median_seconds": round(median(ordered), 3),
        "p95_seconds": round(ordered[rank], 3),
    }


def audit_gemma_timing(path: Path, *, as_of: datetime) -> dict[str, object]:
    """Finite local summary, no article text/URLs or model payloads included."""
    if as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone aware")
    now = as_of.astimezone(UTC)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Existing non-symlink SQLite database required")
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        tables = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        if "impact_attempt_timings" not in tables:
            return {
                "schema_version": "gemma_timing_v1",
                "status": "LEGACY_TIMING_UNAVAILABLE",
                "timed_attempts": 0,
                "sources": [],
                "historical_model_runtime_known": False,
                "note": "Pre-instrumentation checkpoints cannot be split into wait and model time.",
            }
        if "economic_events" not in tables:
            raise ValueError("Economic event timeline required")
        count = db.execute("SELECT COUNT(*) FROM impact_attempt_timings").fetchone()[0]
        if count > MAX_ROWS:
            raise ValueError("Model attempts exceed bounded audit limit")
        rows = db.execute(
            """SELECT e.source,e.processed_at,t.attempted_at,
                      t.provider_returned_at,t.completed_at,
                      t.provider_elapsed_ms,t.processing_elapsed_ms,t.outcome
               FROM impact_attempt_timings t
               JOIN economic_events e ON e.id=t.event_id
               ORDER BY t.id"""
        ).fetchall()
        if len(rows) != count:
            raise ValueError("Orphan timed attempt without economic event")
    groups: dict[str, dict[str, object]] = defaultdict(lambda: {
        "timed_attempts": 0, "successes": 0, "errors": 0,
        "chronology_anomalies": 0, "provider_duration_missing": 0,
        "queue": [], "provider": [], "after_provider": [], "total": [],
    })
    for source, extracted, started, returned, completed, provider_ms, total_ms, outcome in rows:
        group = groups[source]
        group["timed_attempts"] += 1
        if outcome not in {"COMPLETED", "NO_HYPOTHESIS", "ERROR"}:
            raise ValueError("Unknown model attempt outcome")
        group["errors"] += outcome == "ERROR"
        group["successes"] += outcome != "ERROR"
        extraction, begin, end = _time(extracted), _time(started), _time(completed)
        if extraction > begin or begin > end or end > now or total_ms < 0:
            group["chronology_anomalies"] += 1
            continue
        group["queue"].append((begin - extraction).total_seconds())
        group["total"].append(total_ms / 1000)
        if returned is None or provider_ms is None:
            group["provider_duration_missing"] += 1
            continue
        finish = _time(returned)
        if not begin <= finish <= end or provider_ms < 0 or provider_ms > total_ms + 50:
            group["chronology_anomalies"] += 1
            continue
        group["provider"].append(provider_ms / 1000)
        # Includes validation and DB commit, not just raw SQLite I/O.
        group["after_provider"].append(max(0.0, total_ms - provider_ms) / 1000)
    sources = []
    for source, group in sorted(groups.items()):
        sources.append({
            "source": source,
            "timed_attempts": group["timed_attempts"],
            "successes": group["successes"],
            "errors": group["errors"],
            "chronology_anomalies": group["chronology_anomalies"],
            "provider_duration_missing": group["provider_duration_missing"],
            "extraction_to_attempt_wait": _stat(group["queue"]),
            "provider_wall_time": _stat(group["provider"]),
            "validation_and_checkpoint_wall_time": _stat(group["after_provider"]),
            "attempt_total_wall_time": _stat(group["total"]),
        })
    return {
        "schema_version": "gemma_timing_v1",
        "status": "OBSERVED" if rows else "NO_TIMED_ATTEMPTS",
        "timed_attempts": count,
        "sources": sources,
        "as_of_utc": now.isoformat(),
        "historical_model_runtime_known": False,
        "first_token_latency_observed": False,
        "note": (
            "Provider wall time includes local Ollama HTTP, generation and parsing. "
            "The measured extraction-to-attempt gap includes operator queueing and "
            "scheduled delays; old impact_runs have no measured provider duration."
        ),
    }
