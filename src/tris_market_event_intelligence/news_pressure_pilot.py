"""Explicit one-shot, read-only Gemma pressure pilot over existing impact rows.

No change to production checkpoints, launchd or database. Every run is
bounded and reports ungraded or failed signals without claiming accuracy.
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
    EventInput, Impact, ImpactAnalysisError, is_eligible, validate_result,
    ImpactResult,
)
from tris_market_event_intelligence.news_impact_signal import NewsImpactSignal

MAX_PILOT = 3


class PressureAnalyzer(Protocol):
    model: str

    def analyze(
        self, event: EventInput, impact: Impact, *, generated_at: str,
    ) -> NewsImpactSignal: ...


def select_pressure_pilot(db_path: Path, *, limit: int = 3) -> list[tuple[int, EventInput, Impact]]:
    """Select up to three validated prior impact hypotheses using SQLite mode=ro."""
    if type(limit) is not int or not 1 <= limit <= MAX_PILOT:
        raise ValueError("Pilot limit must be 1..3")
    if db_path.is_symlink() or not db_path.is_file():
        raise ValueError("Existing non-symlink news database required")
    with closing(sqlite3.connect(
        db_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5,
    )) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        required = {"economic_events", "market_impacts", "event_metadata"}
        available = {r[0] for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        if not required.issubset(available):
            raise ValueError("Existing impact and event schemas required")
        rows = db.execute(
            """SELECT i.id AS impact_id, i.market, i.sector, i.direction AS impact_direction,
                      i.horizon, i.rationale, i.evidence_excerpt, i.uncertainty,
                      i.drivers_json, i.entities_json,
                      e.id AS event_id, e.classification, e.status,
                      e.direction AS event_direction, e.change_basis_points,
                      e.evidence, e.source, e.source_url, e.detected_at,
                      m.claim_kind, m.verification_note, m.tags_json
               FROM market_impacts i
               JOIN economic_events e ON e.id=i.event_id
               LEFT JOIN event_metadata m ON m.event_id=e.id
               ORDER BY i.id ASC LIMIT 300"""
        ).fetchall()
    selected: list[tuple[int, EventInput, Impact]] = []
    used_events: set[int] = set()
    for row in rows:
        if row["event_id"] in used_events:
            continue
        try:
            event = EventInput(
                event_id=row["event_id"], classification=row["classification"],
                status=row["status"], direction=row["event_direction"],
                change_basis_points=row["change_basis_points"],
                evidence=row["evidence"], source=row["source"],
                source_url=row["source_url"], detected_at=row["detected_at"],
                claim_kind=row["claim_kind"],
                verification_note=row["verification_note"],
                tags=tuple(json.loads(row["tags_json"])) if row["tags_json"] else (),
            )
            impact = Impact(
                market=row["market"], sector=row["sector"],
                direction=row["impact_direction"], horizon=row["horizon"],
                rationale=row["rationale"], evidence_excerpt=row["evidence_excerpt"],
                uncertainty=row["uncertainty"],
                drivers=json.loads(row["drivers_json"]),
                entities=json.loads(row["entities_json"]),
            )
            if not is_eligible(event):
                continue
            validate_result(event, ImpactResult(impacts=[impact]))
        except (ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        selected.append((row["impact_id"], event, impact))
        used_events.add(row["event_id"])
        if len(selected) >= limit:
            break
    return selected


def run_pressure_pilot(
    db_path: Path, *, analyzer: PressureAnalyzer | None,
    limit: int = 3,
) -> dict[str, object]:
    """Offline selection if analyzer=None, explicit local-model calls otherwise."""
    records = select_pressure_pilot(db_path, limit=limit)
    items: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    for impact_id, event, impact in records:
        item: dict[str, object] = {
            "impact_id": impact_id, "event_id": event.event_id,
            "classification": event.classification, "event_status": event.status,
            "market": impact.market, "sector": impact.sector,
            "direction": impact.direction, "horizon": impact.horizon,
        }
        if analyzer is None:
            item["status"] = "PREVIEW_NOT_ANALYZED"
        else:
            started = time.monotonic()
            try:
                signal = analyzer.analyze(
                    event, impact, generated_at=datetime.now(UTC).isoformat()
                )
                if signal.event_id != event.event_id:
                    raise ValueError("Model result event mismatch")
                item.update({
                    "status": "VALIDATED_NEWS_PRESSURE",
                    "intensity": signal.intensity,
                    "intensity_score": signal.intensity_score,
                    "uncertainty": signal.uncertainty,
                    "signal": signal.model_dump(mode="json"),
                })
            except (ImpactAnalysisError, ValueError) as exc:
                item.update({
                    "status": "ANALYSIS_REJECTED",
                    "failure_code": (
                        exc.code if isinstance(exc, ImpactAnalysisError)
                        else "MODEL_OUTPUT_INVALID"
                    ),
                })
            except Exception:
                item.update({
                    "status": "ANALYSIS_FAILED",
                    "failure_code": "MODEL_RUNTIME_FAILURE",
                })
            item["elapsed_ms"] = round((time.monotonic() - started) * 1000, 1)
        counts[item["status"]] += 1
        items.append(item)
    return {
        "schema_version": "news_pressure_pilot_v1",
        "status": (
            "NO_EXISTING_ELIGIBLE_IMPACTS" if not records
            else "PREVIEW_ONLY" if analyzer is None
            else "PILOT_COMPLETED_WITH_REJECTIONS"
            if counts["ANALYSIS_REJECTED"] or counts["ANALYSIS_FAILED"]
            else "PILOT_OUTPUT_VALIDATED"
        ),
        "selected": len(records),
        "limit": limit,
        "counts": dict(sorted(counts.items())),
        "model_id": analyzer.model if analyzer is not None else None,
        "items": items,
        "live_model_quality_verified": False,
        "causal_effect_verified": False,
        "predicts_stock_return": False,
        "production_schedule_changed": False,
        "production_database_modified": False,
        "new_pressure_signals_persisted": 0,
        "note": (
            "A validated format/evidence excerpt is not proof of correct news "
            "intensity. Review each real Gemma output manually before production."
        ),
    }
