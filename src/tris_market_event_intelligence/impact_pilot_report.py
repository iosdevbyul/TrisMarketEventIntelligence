"""Read-only outcome report for an explicitly selected, bounded model pilot.

This is bookkeeping, not a factuality, market accuracy, or causality audit.
"""
from __future__ import annotations

import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

from tris_market_event_intelligence.impact import PROMPT_VERSION
from tris_market_event_intelligence.impact_quality import rationale_quality_code


def parse_event_ids(value: str) -> tuple[int, ...]:
    """Accept only 1–3 explicitly provided unique positive decimal IDs."""
    parts = value.split(",")
    if (
        not 1 <= len(parts) <= 3
        or any(not part.isascii() or not part.isdecimal() for part in parts)
    ):
        raise ValueError("Expected 1..3 comma-separated positive event IDs")
    ids = tuple(int(part) for part in parts)
    if any(item < 1 for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("Event IDs must be unique positive integers")
    return ids


def pilot_outcomes(
    db_path: Path, *, event_ids: tuple[int, ...], model_id: str,
    prompt_version: str = PROMPT_VERSION,
) -> dict[str, object]:
    """Project stored event outcomes without running the model or writing SQLite."""
    if (
        not 1 <= len(event_ids) <= 3
        or any(type(item) is not int or item < 1 for item in event_ids)
        or len(set(event_ids)) != len(event_ids)
        or not model_id.strip()
        or not prompt_version.strip()
    ):
        raise ValueError("Require 1..3 distinct positive IDs, model and prompt version")
    empty = {
        "model_id": model_id, "prompt_version": prompt_version,
        "requested_event_ids": list(event_ids), "items": [],
        "quality_verified": False,
    }
    if not db_path.is_file():
        return {"status": "database_missing", **empty}
    try:
        with closing(sqlite3.connect(
            f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True, timeout=5,
        )) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA query_only=ON")
            tables = {
                row[0] for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if not {"economic_events", "impact_runs", "impact_errors", "market_impacts"}.issubset(
                tables
            ):
                return {"status": "not_initialized", **empty}
            placeholders = ",".join("?" for _ in event_ids)
            events = db.execute(
                f"""SELECT id, classification, status
                    FROM economic_events WHERE id IN ({placeholders})""",
                event_ids,
            ).fetchall()
            runs = db.execute(
                f"""SELECT event_id, outcome FROM impact_runs
                    WHERE model_id=? AND prompt_version=?
                      AND event_id IN ({placeholders})""",
                (model_id, prompt_version, *event_ids),
            ).fetchall()
            errors = db.execute(
                f"""SELECT event_id, last_error, attempts FROM impact_errors
                    WHERE model_id=? AND prompt_version=?
                      AND event_id IN ({placeholders})""",
                (model_id, prompt_version, *event_ids),
            ).fetchall()
            impacts = db.execute(
                f"""SELECT event_id, direction, uncertainty, rationale
                    FROM market_impacts
                    WHERE model_id=? AND prompt_version=?
                      AND event_id IN ({placeholders})""",
                (model_id, prompt_version, *event_ids),
            ).fetchall()
    except sqlite3.Error:
        return {"status": "database_unreadable", **empty}

    event_by_id = {row["id"]: row for row in events}
    run_by_id = {row["event_id"]: row["outcome"] for row in runs}
    error_by_id = {row["event_id"]: row for row in errors}
    impact_by_id: dict[int, list[sqlite3.Row]] = {}
    for impact in impacts:
        impact_by_id.setdefault(impact["event_id"], []).append(impact)

    items: list[dict[str, object]] = []
    outcomes: Counter[str] = Counter()
    failure_counts: Counter[str] = Counter()
    for event_id in event_ids:
        event = event_by_id.get(event_id)
        run = run_by_id.get(event_id)
        error = error_by_id.get(event_id)
        analyses = impact_by_id.get(event_id, [])
        if event is None:
            outcome = "EVENT_NOT_FOUND"
        elif run is not None:
            outcome = run
        elif error is not None:
            outcome = "FAILED_PENDING_RETRY"
        else:
            outcome = "NOT_ATTEMPTED"
        outcomes[outcome] += 1
        if error is not None and run is None:
            failure_counts[error["last_error"]] += 1
        items.append({
            "event_id": event_id,
            "classification": event["classification"] if event else None,
            "event_status": event["status"] if event else None,
            "outcome": outcome,
            "last_error_code": error["last_error"] if error and run is None else None,
            "recorded_failure_attempts": error["attempts"] if error and run is None else 0,
            "stored_impacts": len(analyses),
            "direction_counts": dict(sorted(
                Counter(row["direction"] for row in analyses).items()
            )),
            "uncertainty_counts": dict(sorted(
                Counter(row["uncertainty"] for row in analyses).items()
            )),
            "syntax_warning_counts": dict(sorted(Counter(
                flag for row in analyses
                if (flag := rationale_quality_code(row["rationale"])) is not None
            ).items())),
        })
    return {
        "status": "ok",
        **empty,
        "items": items,
        "outcome_counts": dict(sorted(outcomes.items())),
        "last_failure_counts": dict(sorted(failure_counts.items())),
        "note": (
            "Stored outcomes and syntax flags only; no human semantic review, "
            "model execution or prediction-accuracy claim"
        ),
    }
