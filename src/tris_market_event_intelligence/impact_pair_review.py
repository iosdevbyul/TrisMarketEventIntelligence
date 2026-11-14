"""Private, read-only paired-source review samples for model-version transitions.

This exports model artifacts and original local event evidence; it never judges
if abstention or a directional claim was correct and never calls a model.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

from tris_market_event_intelligence.impact import PROMPT_VERSION


def paired_review_samples(
    db_path: Path, *, model_id: str,
    baseline: str = "market-impact-v7", candidate: str = PROMPT_VERSION,
    limit: int = 5,
) -> dict[str, object]:
    """Return at most five paired, successfully checkpointed cases, not errors."""
    if not model_id.strip() or not baseline.strip() or not candidate.strip():
        raise ValueError("Model and prompt versions are required")
    if baseline == candidate or not 1 <= limit <= 5:
        raise ValueError("Expected distinct versions and a sample limit of 1..5")
    common = {
        "model_id": model_id, "baseline_version": baseline,
        "candidate_version": candidate, "quality_verified": False,
    }
    if not db_path.is_file():
        return {"status": "database_missing", **common, "items": []}
    try:
        with closing(sqlite3.connect(
            f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True, timeout=5,
        )) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA query_only=ON")
            required = {"economic_events", "impact_runs", "market_impacts"}
            tables = {
                row[0] for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if not required.issubset(tables):
                return {"status": "not_initialized", **common, "items": []}
            # Only comparable completed checkpoints, not errors or missing attempts.
            # Prioritize changed decisions then stable decisions, oldest IDs first.
            rows = db.execute(
                """SELECT e.id AS event_id, e.classification, e.status,
                          e.source, e.source_url, e.evidence,
                          old.outcome AS baseline_outcome,
                          new.outcome AS candidate_outcome
                   FROM economic_events e
                   JOIN impact_runs old ON old.event_id=e.id
                     AND old.model_id=? AND old.prompt_version=?
                   JOIN impact_runs new ON new.event_id=e.id
                     AND new.model_id=? AND new.prompt_version=?
                   ORDER BY CASE WHEN old.outcome != new.outcome THEN 0 ELSE 1 END,
                            e.id LIMIT ?""",
                (model_id, baseline, model_id, candidate, limit),
            ).fetchall()
            items = []
            for row in rows:
                hypotheses = db.execute(
                    """SELECT prompt_version, market, sector, direction, horizon,
                              rationale, evidence_excerpt, uncertainty
                       FROM market_impacts WHERE event_id=? AND model_id=?
                         AND prompt_version IN (?, ?)
                       ORDER BY prompt_version, id""",
                    (row["event_id"], model_id, baseline, candidate),
                ).fetchall()
                records = {
                    baseline: [], candidate: [],
                }
                for hypothesis in hypotheses:
                    records[hypothesis["prompt_version"]].append({
                        key: hypothesis[key] for key in (
                            "market", "sector", "direction", "horizon",
                            "rationale", "evidence_excerpt", "uncertainty",
                        )
                    })
                items.append({
                    "event_id": row["event_id"],
                    "classification": row["classification"],
                    "event_status": row["status"],
                    "source": row["source"],
                    "source_url": row["source_url"],
                    "original_evidence": row["evidence"],
                    "baseline_outcome": row["baseline_outcome"],
                    "candidate_outcome": row["candidate_outcome"],
                    "baseline_hypotheses": records[baseline],
                    "candidate_hypotheses": records[candidate],
                    "human_label": "",
                    "human_review_notes": "",
                })
    except sqlite3.Error:
        return {"status": "database_unreadable", **common, "items": []}
    return {
        "status": "ok", **common, "items": items,
        "sampled_events": len(items),
        "note": (
            "Local-only evidence and model text for human review; a candidate "
            "NO_HYPOTHESIS could be justified caution OR a false suppression. "
            "Never classify quality without independent source assessment."
        ),
    }


def save_private_review_bundle(
    folder: Path, filename: str, bundle: dict[str, object],
) -> Path:
    """Create private output, never overwrite or print original evidence."""
    if not filename.startswith("paired-review-") or "/" in filename or "\\" in filename:
        raise ValueError("Invalid review filename")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(bundle, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return path
