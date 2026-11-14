"""Read-only per-event comparison of versioned market impact outcomes.

These counts describe persisted model behavior, not economic correctness,
prediction performance, causal validity or human verification.
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
from tris_market_event_intelligence.impact_quality import rationale_quality_code


def compare_impact_versions(
    db_path: Path, *, model_id: str,
    baseline: str = "market-impact-v7", candidate: str = PROMPT_VERSION,
) -> dict[str, object]:
    """Compare exact model/prompt checkpoints for all eligible distinct events."""
    if not model_id.strip() or not baseline.strip() or not candidate.strip():
        raise ValueError("Model and prompt versions must be nonempty")
    if baseline == candidate:
        raise ValueError("Comparison requires two different prompt versions")
    base: dict[str, object] = {
        "model_id": model_id, "baseline_version": baseline,
        "candidate_version": candidate, "quality_verified": False,
    }
    if not db_path.is_file():
        return {"status": "database_missing", **base}
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
            required = {
                "economic_events", "event_metadata", "impact_runs",
                "impact_errors", "market_impacts",
            }
            if not required.issubset(tables):
                return {"status": "not_initialized", **base}
            events = db.execute(
                """SELECT e.id, e.classification, e.status, m.claim_kind
                   FROM economic_events e
                   LEFT JOIN event_metadata m ON m.event_id=e.id
                   ORDER BY e.id"""
            ).fetchall()
            runs = db.execute(
                """SELECT event_id, prompt_version, outcome
                   FROM impact_runs WHERE model_id=?
                   AND prompt_version IN (?, ?)""",
                (model_id, baseline, candidate),
            ).fetchall()
            errors = db.execute(
                """SELECT event_id, prompt_version, last_error
                   FROM impact_errors WHERE model_id=?
                   AND prompt_version IN (?, ?)""",
                (model_id, baseline, candidate),
            ).fetchall()
            impacts = db.execute(
                """SELECT event_id, prompt_version, rationale
                   FROM market_impacts WHERE model_id=?
                   AND prompt_version IN (?, ?)""",
                (model_id, baseline, candidate),
            ).fetchall()
    except sqlite3.Error:
        return {"status": "database_unreadable", **base}

    checkpoint = {(row["event_id"], row["prompt_version"]): row["outcome"] for row in runs}
    failures = {
        (row["event_id"], row["prompt_version"]): row["last_error"]
        for row in errors
    }
    impact_counts: Counter[tuple[int, str]] = Counter()
    syntax_warnings: Counter[tuple[int, str]] = Counter()
    for row in impacts:
        key = row["event_id"], row["prompt_version"]
        impact_counts[key] += 1
        if rationale_quality_code(row["rationale"]) is not None:
            syntax_warnings[key] += 1

    by_version: dict[str, Counter[str]] = {
        baseline: Counter(), candidate: Counter(),
    }
    transitions: Counter[str] = Counter()
    items: list[dict[str, object]] = []
    for row in events:
        event = EventInput(
            event_id=row["id"], classification=row["classification"],
            status=row["status"], direction=None, change_basis_points=None,
            evidence="", source="", source_url="", detected_at="",
            claim_kind=row["claim_kind"],
        )
        if not is_eligible(event):
            continue
        states: dict[str, str] = {}
        warnings: dict[str, int] = {}
        impacts_per_version: dict[str, int] = {}
        errors_per_version: dict[str, str | None] = {}
        for version in (baseline, candidate):
            key = event.event_id, version
            outcome = checkpoint.get(key)
            error = failures.get(key)
            state = outcome or ("FAILED_PENDING_RETRY" if error else "NOT_ATTEMPTED")
            states[version] = state
            by_version[version][state] += 1
            warnings[version] = syntax_warnings[key]
            impacts_per_version[version] = impact_counts[key]
            errors_per_version[version] = error if outcome is None else None
        transition = f"{states[baseline]} -> {states[candidate]}"
        transitions[transition] += 1
        items.append({
            "event_id": event.event_id, "classification": event.classification,
            "event_status": event.status,
            "baseline_outcome": states[baseline],
            "candidate_outcome": states[candidate],
            "baseline_impacts": impacts_per_version[baseline],
            "candidate_impacts": impacts_per_version[candidate],
            "baseline_syntax_warnings": warnings[baseline],
            "candidate_syntax_warnings": warnings[candidate],
            "baseline_last_error": errors_per_version[baseline],
            "candidate_last_error": errors_per_version[candidate],
        })

    return {
        "status": "ok", **base, "eligible_events": len(items),
        "outcome_counts": {
            version: dict(sorted(by_version[version].items()))
            for version in (baseline, candidate)
        },
        "transition_counts": dict(sorted(transitions.items())),
        "items": items,
        "note": (
            "Read-only status transitions and syntax warnings; "
            "NO_HYPOTHESIS is not automatically better or worse than COMPLETED. "
            "Source support, market causality and forecast skill require human review."
        ),
    }
