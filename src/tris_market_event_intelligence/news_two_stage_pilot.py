"""Explicit two-stage news-only pilot using captured, eligible news events.

Read-only SQLite selection -> existing evidence-bound direction model -> optional
source-grounded pressure model. No impact checkpoints, news DB writes, or
price feeds. This is a quality pilot, never the production collector.
"""
from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from tris_market_event_intelligence.impact import (
    EventInput,
    Impact,
    ImpactResult,
    is_eligible,
    validate_result,
)
from tris_market_event_intelligence.impact_pilot import pilot_candidates
from tris_market_event_intelligence.news_impact_signal import NewsImpactSignal

MAX_EVENTS = 3


class DirectionModel(Protocol):
    model_id: str

    def analyze(self, event: EventInput) -> ImpactResult: ...


class StrengthModel(Protocol):
    model: str

    def analyze(
        self, event: EventInput, impact: Impact, *, generated_at: str,
    ) -> NewsImpactSignal: ...


def select_news_events(
    path: Path, *, limit: int = MAX_EVENTS, model_id: str = "gemma2:latest",
) -> list[EventInput]:
    """Select only eligible, not-already-checkpointed distinct real events."""
    if type(limit) is not int or not 1 <= limit <= MAX_EVENTS:
        raise ValueError("Pilot limit must be 1..3")
    if path.is_symlink() or not path.is_file():
        raise ValueError("Existing non-symlink DB required")
    candidates = pilot_candidates(
        path, model_id=model_id, limit=limit, new_only=True,
    )
    if candidates["status"] != "ok":
        raise ValueError("Existing impact-pilot schema unavailable")
    ids = [item["event_id"] for item in candidates["candidates"]]
    if not ids:
        return []
    placeholders = ",".join("?" for _ in ids)
    with closing(sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5,
    )) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        rows = db.execute(
            f"""SELECT e.id, e.classification, e.status, e.direction,
                       e.change_basis_points, e.evidence, e.source, e.source_url,
                       e.detected_at, m.claim_kind, m.verification_note, m.tags_json
                FROM economic_events e
                LEFT JOIN event_metadata m ON m.event_id=e.id
                WHERE e.id IN ({placeholders})""", ids,
        ).fetchall()
    by_id: dict[int, EventInput] = {}
    for row in rows:
        try:
            event = EventInput(
                event_id=row["id"], classification=row["classification"],
                status=row["status"], direction=row["direction"],
                change_basis_points=row["change_basis_points"],
                evidence=row["evidence"], source=row["source"],
                source_url=row["source_url"], detected_at=row["detected_at"],
                claim_kind=row["claim_kind"],
                verification_note=row["verification_note"],
                tags=tuple(json.loads(row["tags_json"])) if row["tags_json"] else (),
            )
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if is_eligible(event) and event.evidence and event.source_url:
            by_id[event.event_id] = event
    return [by_id[i] for i in ids if i in by_id]


def run_two_stage_pilot(
    path: Path, *, direction_model: DirectionModel | None,
    strength_model: StrengthModel | None, limit: int = MAX_EVENTS,
    model_id: str = "gemma2:latest",
) -> dict[str, object]:
    """Preview when both models are None; otherwise explicitly call both.

    Reject mixed operation to prevent a direction result silently bypassing
    pressure validation. Each input event is selected from real SQLite.
    """
    if (direction_model is None) != (strength_model is None):
        raise ValueError("Direction and strength models must be enabled together")
    if direction_model is not None and (
        direction_model.model_id != model_id or strength_model.model != model_id
    ):
        raise ValueError("Pilot model IDs must match requested model")
    events = select_news_events(path, limit=limit, model_id=model_id)
    counts: Counter[str] = Counter()
    direction_calls = 0
    strength_calls = 0
    items: list[dict[str, object]] = []
    for event in events:
        entry: dict[str, object] = {
            "event_id": event.event_id,
            "classification": event.classification,
            "event_status": event.status,
            "source_url": event.source_url,
        }
        if direction_model is None:
            entry["status"] = "PREVIEW_NOT_ANALYZED"
        else:
            start = time.monotonic()
            direction_calls += 1
            try:
                result = validate_result(event, direction_model.analyze(event))
                # Multiple defensible sectors may be returned, but keep calls
                # bounded by one chosen directional hypothesis per article.
                directional = next(
                    (impact for impact in result.impacts
                     if impact.direction in {"POSITIVE", "NEGATIVE", "MIXED"}),
                    None,
                )
                if directional is None:
                    entry["status"] = "NO_SUPPORTED_DIRECTION"
                else:
                    strength_calls += 1
                    signal = strength_model.analyze(
                        event, directional, generated_at=datetime.now(UTC).isoformat(),
                    )
                    if (
                        signal.event_id != event.event_id
                        or signal.direction != directional.direction
                        or signal.market != directional.market
                        or signal.sector != directional.sector
                    ):
                        raise ValueError("Cross-stage provenance mismatch")
                    entry.update({
                        "status": "TWO_STAGE_OUTPUT_VALIDATED",
                        "signal": signal.model_dump(mode="json"),
                    })
            except Exception as exc:  # noqa: BLE001
                # Keep model errors out of user-visible raw text or logs.
                from tris_market_event_intelligence.impact import ImpactAnalysisError

                entry["status"] = "REJECTED_OR_FAILED"
                entry["failure_code"] = (
                    exc.code if isinstance(exc, ImpactAnalysisError)
                    else "INVALID_OR_FAILED_MODEL_RESPONSE"
                )
            entry["elapsed_ms"] = round((time.monotonic() - start) * 1000, 1)
        counts[entry["status"]] += 1
        items.append(entry)
    return {
        "schema_version": "news_two_stage_pilot_v1",
        "status": (
            "NO_ELIGIBLE_NEW_EVENTS" if not events
            else "PREVIEW_ONLY" if direction_model is None
            else "PILOT_WITH_FAILURES" if counts["REJECTED_OR_FAILED"]
            else "TWO_STAGE_PILOT_FINISHED"
        ),
        "selected_events": len(events),
        "direction_model_calls": direction_calls,
        "strength_model_calls": strength_calls,
        "model_id": model_id,
        "items": items,
        "counts": dict(sorted(counts.items())),
        "operational_quality_verified": False,
        "stock_returns_predicted": False,
        "production_database_modified": False,
        "production_schedule_changed": False,
        "note": (
            "Model outputs are conditional hypotheses, not validated financial "
            "effects. No input events are manufactured or checkpointed."
        ),
    }
