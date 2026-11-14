"""Read-only event lifecycle timing audit over existing local SQLite records.

Do not equate model run checkpoint time with first token or first inference.
Reports trusted only as observed local timestamps, not source-published truth.
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

_MAX_EVENTS = 50_000


def _instant(value: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError("Lifecycle timestamps must be strings")
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("Malformed lifecycle timestamp") from exc
    if timestamp.utcoffset() is None:
        raise ValueError("Lifecycle timestamps require explicit time zones")
    return timestamp.astimezone(UTC)


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median_seconds": None, "p95_seconds": None}
    ordered = sorted(values)
    # Nearest-rank p95 so outliers are not ignored for small samples.
    rank = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return {
        "count": len(values),
        "median_seconds": round(median(values), 3),
        "p95_seconds": round(ordered[rank], 3),
    }


def audit_event_lifecycle(
    database: Path, *, as_of: datetime, maximum_events: int = _MAX_EVENTS,
) -> dict[str, object]:
    """Audit existing persisted timestamps without initialization or writes."""
    if database.is_symlink() or not database.is_file():
        raise ValueError("Existing non-symlink SQLite database required")
    if type(maximum_events) is not int or not 1 <= maximum_events <= _MAX_EVENTS:
        raise ValueError("maximum_events must be 1..50000")
    if not isinstance(as_of, datetime) or as_of.utcoffset() is None:
        raise ValueError("as_of requires a timezone")
    audit_at = as_of.astimezone(UTC)
    with closing(sqlite3.connect(
        database.resolve().as_uri() + "?mode=ro", uri=True,
    )) as db:
        db.execute("PRAGMA query_only=ON")
        tables = {
            row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if not {"articles", "economic_events"}.issubset(tables):
            raise ValueError("News article and economic event tables required")
        has_runs = "impact_runs" in tables
        count = db.execute("SELECT COUNT(*) FROM economic_events").fetchone()[0]
        if count > maximum_events:
            raise ValueError("Event count exceeds bounded audit limit")
        events = db.execute(
            "SELECT e.id,a.source,a.detected_at,e.detected_at,"
            "e.processed_at,e.published_at "
            "FROM economic_events e "
            "JOIN articles a ON a.id=e.article_id ORDER BY e.id"
        ).fetchall()
        runs = []
        if has_runs:
            run_count = db.execute("SELECT COUNT(*) FROM impact_runs").fetchone()[0]
            if run_count > maximum_events * 5:
                raise ValueError("Impact run count exceeds bounded audit limit")
            runs = db.execute(
                "SELECT event_id,model_id,prompt_version,processed_at,outcome "
                "FROM impact_runs ORDER BY event_id,model_id,prompt_version"
            ).fetchall()

    by_event: dict[int, list[tuple[str, str, datetime, str]]] = defaultdict(list)
    for event_id, model, prompt, completed, outcome in runs:
        if outcome not in {"COMPLETED", "NO_HYPOTHESIS"}:
            raise ValueError("Unknown model checkpoint outcome")
        timestamp = _instant(completed)
        if timestamp > audit_at:
            raise ValueError("Analysis checkpoint occurs after audit time")
        by_event[event_id].append((model, prompt, timestamp, outcome))

    groups: dict[str, dict[str, object]] = {}

    def group_for(source: str) -> dict[str, object]:
        return groups.setdefault(source, {
            "economic_events": 0, "checkpointed_events": 0,
            "uncheckpointed_events": 0,
            "model_checkpoints": 0, "no_hypothesis_checkpoints": 0,
            "chronology_anomalies": 0,
            "first_seen_to_extraction": [],
            "extraction_to_checkpoint": [],
            "first_seen_to_checkpoint": [],
            "published_to_checkpoint": [],
            "publication_missing": 0,
            "publication_after_first_seen": 0,
        })

    for event_id, source, article_seen, event_seen, extracted, publication in events:
        group = group_for(source)
        group["economic_events"] += 1
        seen = _instant(article_seen)
        copied_seen = _instant(event_seen)
        extraction = _instant(extracted)
        related = by_event.pop(event_id, [])
        group["model_checkpoints"] += len(related)
        group["no_hypothesis_checkpoints"] += sum(
            outcome == "NO_HYPOTHESIS" for _, _, _, outcome in related
        )
        if related:
            group["checkpointed_events"] += 1
        else:
            group["uncheckpointed_events"] += 1
        # Old records may predate precise HTTP-response instrumentation.
        # Reject inconsistencies from latency distributions, do not repair.
        if seen != copied_seen or seen > extraction or extraction > audit_at:
            group["chronology_anomalies"] += 1
            continue
        group["first_seen_to_extraction"].append((extraction - seen).total_seconds())
        if publication is None:
            group["publication_missing"] += 1
            published = None
        else:
            published = _instant(publication)
            if published > seen:
                group["publication_after_first_seen"] += 1
                published = None
        for _, _, completed, _ in related:
            if completed < extraction:
                group["chronology_anomalies"] += 1
                continue
            group["extraction_to_checkpoint"].append(
                (completed - extraction).total_seconds()
            )
            group["first_seen_to_checkpoint"].append(
                (completed - seen).total_seconds()
            )
            if published is not None:
                group["published_to_checkpoint"].append(
                    (completed - published).total_seconds()
                )
    if by_event:
        raise ValueError("Orphan impact checkpoint references unknown economic event")

    sources = []
    for source, group in sorted(groups.items()):
        distributions = {
            name + "_latency": _distribution(group.pop(name))
            for name in (
                "first_seen_to_extraction", "extraction_to_checkpoint",
                "first_seen_to_checkpoint", "published_to_checkpoint",
            )
        }
        sources.append({"source": source, **group, **distributions})
    return {
        "schema_version": "event_lifecycle_latency_v1",
        "as_of_utc": audit_at.isoformat(),
        "source_count": len(sources),
        "economic_events": len(events),
        "model_checkpoint_table_available": has_runs,
        "model_checkpoints": len(runs),
        "sources": sources,
        "news_to_usable_inference_verified": False,
        "real_time_detection_verified": False,
        "publisher_timestamp_authenticated": False,
        "gemma_token_start_observed": False,
        "live_market_coverage_verified": False,
        "note": (
            "RSS receipt -> event extraction -> persisted model checkpoint timings. "
            "Checkpoint is AFTER model work and DB persistence, not first usable token. "
            "Legacy receipt timestamps may represent HTTP request start. "
            "Missing checkpoints mean missing latency data, not zero seconds."
        ),
    }
