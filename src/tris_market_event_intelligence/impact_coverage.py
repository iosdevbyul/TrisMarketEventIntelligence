"""Read-only model coverage census over collected economic events.

Counts are not quality certification and do not run local inference.
"""
from __future__ import annotations

import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

from tris_market_event_intelligence.impact import (
    PROMPT_VERSION,
    EventInput,
    is_eligible,
)


def impact_coverage(
    db_path: Path, *, model_id: str, prompt_version: str = PROMPT_VERSION,
    sample_limit: int = 10,
) -> dict[str, object]:
    """Count distinct eligible events, versioned checkpoints, and failure codes."""
    if not model_id.strip() or not prompt_version.strip():
        raise ValueError("model_id and prompt_version must be nonempty")
    if not 1 <= sample_limit <= 100:
        raise ValueError("sample_limit must be 1..100")
    if not db_path.is_file():
        return {"status": "database_missing", "quality_verified": False}
    try:
        with closing(sqlite3.connect(
            f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True, timeout=5,
        )) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA query_only=ON")
            tables = {
                r[0] for r in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            required = {"economic_events", "impact_runs", "market_impacts", "impact_errors"}
            if not required.issubset(tables):
                return {"status": "not_initialized", "quality_verified": False}
            metadata = "event_metadata" in tables
            join = (
                "LEFT JOIN event_metadata m ON m.event_id=e.id"
                if metadata else ""
            )
            claim_col = "m.claim_kind" if metadata else "NULL"
            rows = db.execute(
                f"""SELECT e.id, e.classification, e.status,
                           {claim_col} AS claim_kind
                    FROM economic_events e {join}
                    ORDER BY e.id"""
            ).fetchall()
            eligible_ids = []
            by_category: Counter[str] = Counter()
            by_status: Counter[str] = Counter()
            for row in rows:
                event = EventInput(
                    event_id=row["id"], classification=row["classification"],
                    status=row["status"], direction=None, change_basis_points=None,
                    evidence="", source="", source_url="", detected_at="",
                    claim_kind=row["claim_kind"],
                )
                if is_eligible(event):
                    eligible_ids.append(row["id"])
                    by_category[row["classification"]] += 1
                    by_status[row["status"]] += 1
            run_rows = db.execute(
                """SELECT event_id, outcome FROM impact_runs
                   WHERE model_id=? AND prompt_version=?""",
                (model_id, prompt_version),
            ).fetchall()
            runs = {r["event_id"]: r["outcome"] for r in run_rows}
            impact_rows = db.execute(
                """SELECT i.event_id, i.direction, i.uncertainty
                   FROM market_impacts i
                   WHERE i.model_id=? AND i.prompt_version=?""",
                (model_id, prompt_version),
            ).fetchall()
            errors = db.execute(
                """SELECT event_id, last_error, attempts FROM impact_errors
                   WHERE model_id=? AND prompt_version=?""",
                (model_id, prompt_version),
            ).fetchall()
            all_versions = db.execute(
                """SELECT prompt_version, outcome, COUNT(*) AS n FROM impact_runs
                   WHERE model_id=? GROUP BY prompt_version, outcome
                   ORDER BY prompt_version, outcome""",
                (model_id,),
            ).fetchall()
    except sqlite3.Error:
        return {"status": "database_unreadable", "quality_verified": False}

    pending = [event_id for event_id in eligible_ids if event_id not in runs]
    eligible_set = set(eligible_ids)
    errors_by_code = Counter(
        row["last_error"] for row in errors if row["event_id"] in eligible_set
    )
    failures = {row["event_id"] for row in errors if row["event_id"] in eligible_set}
    return {
        "status": "ok",
        "model_id": model_id, "prompt_version": prompt_version,
        "total_events": len(rows), "eligible_events": len(eligible_ids),
        "ineligible_events": len(rows) - len(eligible_ids),
        "eligible_by_category": dict(sorted(by_category.items())),
        "eligible_by_status": dict(sorted(by_status.items())),
        "completed_events": sum(
            runs.get(event_id) == "COMPLETED" for event_id in eligible_ids
        ),
        "no_hypothesis_events": sum(
            runs.get(event_id) == "NO_HYPOTHESIS" for event_id in eligible_ids
        ),
        "pending_events": len(pending),
        "pending_with_recorded_failure": len(failures.intersection(pending)),
        "pending_sample_event_ids": pending[:sample_limit],
        "stored_impacts": len(impact_rows),
        "direction_counts": dict(sorted(Counter(r["direction"] for r in impact_rows).items())),
        "uncertainty_counts": dict(sorted(
            Counter(r["uncertainty"] for r in impact_rows).items()
        )),
        "last_failure_counts": dict(sorted(errors_by_code.items())),
        "historical_run_counts": [
            dict(row) for row in all_versions
        ],
        "quality_verified": False,
        "note": "Read-only coverage counts, not verified economic truth or model accuracy",
    }
