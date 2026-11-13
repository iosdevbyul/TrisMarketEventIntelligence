"""Independent review of saved model hypotheses against original economic events.

Only humans can label causal validity or direction justification. No model calls,
no SQLite writes, and no automatic acceptance based on a text-matching heuristic.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

from tris_market_event_intelligence.impact_quality import rationale_quality_code

DIMENSIONS = {
    "source_support": ("SUPPORTED", "UNSUPPORTED", "INDETERMINATE"),
    "causal_reasoning": ("JUSTIFIED", "UNJUSTIFIED", "INDETERMINATE"),
    "direction_reasoning": ("JUSTIFIED", "UNJUSTIFIED", "INDETERMINATE"),
    "uncertainty_language": ("CONDITIONAL", "OVERSTATED", "INDETERMINATE"),
    "sentence_completion": ("COMPLETE", "INCOMPLETE", "INDETERMINATE"),
}
COLUMNS = (
    "impact_id", "event_id", "model_id", "prompt_version", "event_status",
    "claim_kind", "source", "source_url", "capture_sha256",
    "original_evidence", "rationale", "evidence_excerpt", "market",
    "sector", "direction", "horizon", "uncertainty", "snapshot_sha256",
    "automated_quality_flag", *DIMENSIONS, "review_notes",
)
SNAPSHOT_FIELDS = COLUMNS[:17]


def _connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.is_file():
        raise FileNotFoundError("No local event database; no file was created")
    db = sqlite3.connect(
        f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True, timeout=5,
    )
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    return db


def _snapshot(record: dict[str, object]) -> str:
    payload = [record.get(field) for field in SNAPSHOT_FIELDS]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _records(db: sqlite3.Connection, model_id: str | None) -> list[dict[str, object]]:
    rows = db.execute(
        """SELECT i.id AS impact_id, i.event_id, i.model_id, i.prompt_version,
                  e.status AS event_status, m.claim_kind, e.source, e.source_url,
                  e.capture_sha256, e.evidence AS original_evidence,
                  i.rationale, i.evidence_excerpt, i.market, i.sector,
                  i.direction, i.horizon, i.uncertainty
           FROM market_impacts i
           JOIN economic_events e ON e.id=i.event_id
           LEFT JOIN event_metadata m ON m.event_id=i.event_id
           WHERE (? IS NULL OR i.model_id=?)
           ORDER BY i.event_id, i.id""",
        (model_id, model_id),
    ).fetchall()
    return [dict(row) for row in rows]


def export_impact_review(
    db_path: Path, output: Path, *, count: int = 30, model_id: str | None = None,
) -> dict[str, object]:
    """Deterministically pick distinct events first; export to local CSV only."""
    if not 1 <= count <= 1000:
        raise ValueError("count must be 1..1000")
    with closing(_connect(db_path)) as db:
        records = _records(db, model_id)
    selected: list[dict[str, object]] = []
    seen_events: set[int] = set()
    for row in records:
        if row["event_id"] not in seen_events:
            selected.append(row)
            seen_events.add(row["event_id"])
        if len(selected) >= count:
            break
    # Include additional hypotheses only when fewer distinct events exist.
    for row in records:
        if len(selected) >= count:
            break
        if row not in selected:
            selected.append(row)
    if not selected:
        return {
            "mode": "UNLABELLED_IMPACT_REVIEW", "available_impacts": 0,
            "exported_impacts": 0, "distinct_events": 0, "reviewed_impacts": 0,
            "quality_verified": False,
        }
    if output.exists():
        raise FileExistsError("Review sheet already exists; refusing overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for row in selected:
            writer.writerow({
                **row, "snapshot_sha256": _snapshot(row),
                "automated_quality_flag": rationale_quality_code(str(row["rationale"])) or "NONE",
                **{dimension: "" for dimension in DIMENSIONS},
                "review_notes": "",
            })
    return {
        "mode": "UNLABELLED_IMPACT_REVIEW",
        "available_impacts": len(records), "exported_impacts": len(selected),
        "distinct_events": len({r["event_id"] for r in selected}),
        "reviewed_impacts": 0, "quality_verified": False,
    }


def score_impact_review(db_path: Path, sheet: Path) -> dict[str, object]:
    """Score complete human labels only; never infer 'gold' from the model itself."""
    with sheet.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not set(COLUMNS).issubset(reader.fieldnames or ()):
            raise ValueError("Review sheet is missing required fields")
        entries = list(reader)
    if not entries:
        raise ValueError("Empty review sheet")
    ids: list[int] = []
    for row in entries:
        try:
            identifier = int(row["impact_id"])
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid impact ID") from exc
        if identifier <= 0 or identifier in ids:
            raise ValueError("Duplicate or invalid impact ID")
        ids.append(identifier)
    with closing(_connect(db_path)) as db:
        placeholders = ",".join("?" for _ in ids)
        records = db.execute(
            f"""SELECT i.id AS impact_id, i.event_id, i.model_id, i.prompt_version,
                      e.status AS event_status, m.claim_kind, e.source, e.source_url,
                      e.capture_sha256, e.evidence AS original_evidence,
                      i.rationale, i.evidence_excerpt, i.market, i.sector,
                      i.direction, i.horizon, i.uncertainty
                FROM market_impacts i JOIN economic_events e ON e.id=i.event_id
                LEFT JOIN event_metadata m ON m.event_id=i.event_id
                WHERE i.id IN ({placeholders})""", ids,
        ).fetchall()
    by_id = {int(row["impact_id"]): dict(row) for row in records}
    if len(by_id) != len(entries):
        raise ValueError("A reviewed impact is no longer available")
    counts: dict[str, Counter[str]] = {name: Counter() for name in DIMENSIONS}
    reviewed = 0
    pending = 0
    reviewed_ids: list[int] = []
    for row, identifier in zip(entries, ids, strict=True):
        source = by_id[identifier]
        if row["snapshot_sha256"] != _snapshot(source) or any(
            row[field] != str(source[field]) for field in SNAPSHOT_FIELDS
        ):
            raise ValueError("Review sheet no longer matches original immutable analysis")
        expected_flag = rationale_quality_code(str(source["rationale"])) or "NONE"
        if row["automated_quality_flag"] != expected_flag:
            raise ValueError("Automated quality flag was modified")
        labels = [row[dimension].strip() for dimension in DIMENSIONS]
        if not any(labels):
            pending += 1
            continue
        if any(not label for label in labels):
            raise ValueError("Partially labelled review; complete all five dimensions")
        for dimension, label in zip(DIMENSIONS, labels, strict=True):
            if label not in DIMENSIONS[dimension]:
                raise ValueError("Unknown human review label")
            counts[dimension][label] += 1
        reviewed += 1
        reviewed_ids.append(identifier)
    return {
        "mode": "INDEPENDENT_HUMAN_IMPACT_REVIEW",
        "total_sheet_impacts": len(entries), "reviewed_impacts": reviewed,
        "pending_impacts": pending, "reviewed_ids": reviewed_ids,
        "dimension_counts": {key: dict(counts[key]) for key in DIMENSIONS},
        "quality_verified": False,
        "note": "Descriptive human labels only; no automatic market-quality certification",
    }
