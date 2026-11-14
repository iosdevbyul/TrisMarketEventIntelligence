"""Read-only, bounded selection of distinct eligible events for manual Ollama pilots."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from tris_market_event_intelligence.impact_candidate_hygiene import (
    article_key,
    evidence_key,
    insufficient_evidence,
)
from tris_market_event_intelligence.impact import (
    PROMPT_VERSION,
    EventInput,
    is_eligible,
)


def pilot_candidates(
    db_path: Path, *, model_id: str, limit: int = 3,
    prompt_version: str = PROMPT_VERSION, new_only: bool = False,
) -> dict[str, object]:
    """Favor not-yet-attempted events across categories; never run inference."""
    if not model_id.strip() or not prompt_version.strip():
        raise ValueError("model_id and prompt_version must be nonempty")
    if not 1 <= limit <= 3:
        raise ValueError("pilot limit must be 1..3")
    if not db_path.is_file():
        return {"status": "database_missing", "candidates": [], "quality_verified": False}
    try:
        with closing(sqlite3.connect(
            f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True, timeout=5,
        )) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA query_only=ON")
            required = {"economic_events", "event_metadata", "impact_runs", "impact_errors"}
            available = {
                row[0] for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if not required.issubset(available):
                return {
                    "status": "not_initialized", "candidates": [],
                    "quality_verified": False,
                }
            rows = db.execute(
                """SELECT e.id, e.classification, e.status, e.source_url,
                          e.evidence, m.claim_kind, r.outcome, err.last_error,
                          EXISTS (
                              SELECT 1 FROM impact_runs old
                              WHERE old.event_id=e.id AND old.model_id=?
                          ) AS previously_completed_other_version,
                          EXISTS (
                              SELECT 1 FROM impact_errors old_error
                              WHERE old_error.event_id=e.id AND old_error.model_id=?
                          ) AS previously_failed_any_version
                   FROM economic_events e
                   LEFT JOIN event_metadata m ON m.event_id=e.id
                   LEFT JOIN impact_runs r ON r.event_id=e.id AND r.model_id=?
                       AND r.prompt_version=?
                   LEFT JOIN impact_errors err ON err.event_id=e.id
                       AND err.model_id=? AND err.prompt_version=?
                   ORDER BY e.id""",
                (model_id, model_id, model_id, prompt_version, model_id, prompt_version),
            ).fetchall()
    except sqlite3.Error:
        return {"status": "database_unreadable", "candidates": [], "quality_verified": False}

    fresh: list[dict[str, object]] = []
    failed: list[dict[str, object]] = []
    eligible = 0
    completed = 0
    skipped_duplicate = 0
    skipped_sparse = 0
    # Prevent selecting a second copy of a previously attempted article,
    # irrespective of its source feed or prompt version.
    used_urls: set[str] = set()
    used_evidence: set[str] = set()
    if new_only:
        for record in rows:
            if record["previously_completed_other_version"] or record["previously_failed_any_version"]:
                url = article_key(record["source_url"])
                evidence = evidence_key(record["evidence"])
                if url:
                    used_urls.add(url)
                if evidence:
                    used_evidence.add(evidence)
    for row in rows:
        event = EventInput(
            event_id=row["id"], classification=row["classification"],
            status=row["status"], direction=None, change_basis_points=None,
            evidence="", source="", source_url="", detected_at="",
            claim_kind=row["claim_kind"],
        )
        if not is_eligible(event):
            continue
        eligible += 1
        if row["outcome"] is not None:
            completed += 1
            continue
        if new_only and (
            row["previously_completed_other_version"] or row["previously_failed_any_version"]
        ):
            continue
        if new_only:
            url = article_key(row["source_url"])
            evidence = evidence_key(row["evidence"])
            if url and url in used_urls or evidence and evidence in used_evidence:
                skipped_duplicate += 1
                continue
            if insufficient_evidence(row["classification"], row["evidence"]):
                skipped_sparse += 1
                continue
            if url:
                used_urls.add(url)
            if evidence:
                used_evidence.add(evidence)
        item = {
            "event_id": row["id"], "classification": row["classification"],
            "status": row["status"],
            "previously_analyzed": bool(row["previously_completed_other_version"]),
            "prior_failure": row["last_error"] is not None,
        }
        (failed if item["prior_failure"] else fresh).append(item)

    # Round robin by category avoids a pilot dominated by the smallest IDs
    # from a single type of news. Prior failed events are NOT auto-selected.
    selected: list[dict[str, object]] = []
    seen_categories: set[str] = set()
    for item in fresh:
        if item["classification"] not in seen_categories:
            selected.append(item)
            seen_categories.add(str(item["classification"]))
            if len(selected) == limit:
                break
    if len(selected) < limit:
        selected_ids = {item["event_id"] for item in selected}
        selected.extend(
            item for item in fresh
            if item["event_id"] not in selected_ids
        )
    selected = selected[:limit]
    return {
        "status": "ok", "model_id": model_id, "prompt_version": prompt_version,
        "new_only": new_only,
        "eligible_events": eligible,
        "already_checkpointed_events": completed,
        "fresh_pending_events": len(fresh),
        "pending_with_prior_failure": len(failed),
        "skipped_duplicate_sources": skipped_duplicate,
        "skipped_sparse_evidence": skipped_sparse,
        "candidates": selected,
        "quality_verified": False,
        "note": "Read-only cohort suggestion; explicit model invocation is required",
    }
