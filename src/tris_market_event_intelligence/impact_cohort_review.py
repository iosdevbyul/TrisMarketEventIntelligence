"""Private source-linked review of a bounded, newly analyzed v8 cohort."""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

from tris_market_event_intelligence.impact import PROMPT_VERSION
from tris_market_event_intelligence.impact_pilot_report import parse_event_ids


def cohort_review(
    db_path: Path, *, event_ids: tuple[int, ...], model_id: str,
    prompt_version: str = PROMPT_VERSION,
) -> dict[str, object]:
    """Read originals and exact-version hypotheses; never invoke a provider."""
    parse_event_ids(",".join(str(i) for i in event_ids))
    if not model_id.strip() or not prompt_version.strip():
        raise ValueError("Model ID and prompt version must be nonempty")
    common: dict[str, object] = {
        "model_id": model_id, "prompt_version": prompt_version,
        "selected_event_ids": list(event_ids), "quality_verified": False,
    }
    if not db_path.is_file():
        return {"status": "database_missing", **common, "items": []}
    try:
        with closing(sqlite3.connect(
            f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True, timeout=5,
        )) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA query_only=ON")
            available = {
                row[0] for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            if not {"economic_events", "impact_runs", "market_impacts"}.issubset(available):
                return {"status": "not_initialized", **common, "items": []}
            items: list[dict[str, object]] = []
            for event_id in event_ids:
                event = db.execute(
                    """SELECT id, classification, status, evidence, source, source_url
                       FROM economic_events WHERE id=?""",
                    (event_id,),
                ).fetchone()
                if event is None:
                    continue
                run = db.execute(
                    """SELECT outcome FROM impact_runs
                       WHERE event_id=? AND model_id=? AND prompt_version=?""",
                    (event_id, model_id, prompt_version),
                ).fetchone()
                errors = db.execute(
                    """SELECT last_error, attempts FROM impact_errors
                       WHERE event_id=? AND model_id=? AND prompt_version=?""",
                    (event_id, model_id, prompt_version),
                ).fetchone() if "impact_errors" in available else None
                hypotheses = db.execute(
                    """SELECT market, sector, direction, horizon, rationale,
                              evidence_excerpt, uncertainty, drivers_json, entities_json
                       FROM market_impacts WHERE event_id=? AND model_id=?
                         AND prompt_version=? ORDER BY id""",
                    (event_id, model_id, prompt_version),
                ).fetchall()
                items.append({
                    "event_id": event_id,
                    "classification": event["classification"],
                    "event_status": event["status"],
                    "source": event["source"],
                    "source_url": event["source_url"],
                    "original_evidence": event["evidence"],
                    "outcome": run["outcome"] if run else (
                        "FAILED_PENDING_RETRY" if errors else "NOT_ATTEMPTED"
                    ),
                    "last_error_code": errors["last_error"] if errors and not run else None,
                    "hypotheses": [
                        {
                            **{field: row[field] for field in (
                                "market", "sector", "direction", "horizon",
                                "rationale", "evidence_excerpt", "uncertainty",
                            )},
                            "drivers": json.loads(row["drivers_json"]),
                            "entities": json.loads(row["entities_json"]),
                        }
                        for row in hypotheses
                    ],
                    "human_label": "",
                    "human_review_notes": "",
                })
    except (sqlite3.Error, ValueError, TypeError):
        return {"status": "database_unreadable", **common, "items": []}
    return {
        "status": "ok", **common, "items": items,
        "note": (
            "Private source-linked unlabelled review. Not human verified; "
            "an empty hypothesis may be justified caution or false suppression."
        ),
    }


def save_cohort_review(
    folder: Path, filename: str, data: dict[str, object],
) -> Path:
    """Create owner-only JSON once; do not overwrite or print private evidence."""
    if not filename.startswith("cohort-review-") or "/" in filename or "\\" in filename:
        raise ValueError("Invalid review filename")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return path
