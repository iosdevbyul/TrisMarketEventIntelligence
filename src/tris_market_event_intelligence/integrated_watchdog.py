"""Finite read-only watchdog for explicitly scheduled integrated news cycles."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.operations import OPERATIONAL_SOURCES

MAX_STATUS_BYTES = 65_536


def audit_integrated_schedule(
    status_path: Path, *, as_of: datetime,
    max_age_seconds: int = 1800,
    expected_sources: tuple[str, ...] = OPERATIONAL_SOURCES,
) -> dict[str, object]:
    """Read last-run checkpoint only; does not install or run any services."""
    if not isinstance(as_of, datetime) or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    if type(max_age_seconds) is not int or not 1 <= max_age_seconds <= 86400:
        raise ValueError("max_age_seconds must be 1..86400")
    if not expected_sources or len(set(expected_sources)) != len(expected_sources) or (
        set(expected_sources) - set(OPERATIONAL_SOURCES)
    ):
        raise ValueError("Unknown or duplicate operational sources")
    report: dict[str, object] = {
        "schema_version": "integrated_schedule_watchdog_v1",
        "as_of_utc": as_of.astimezone(UTC).isoformat(),
        "status": "BLOCKED",
        "reason": "NO_COMPLETED_INTEGRATED_RUN",
        "max_age_seconds": max_age_seconds,
        "sources_expected": list(expected_sources),
        "model_inference_verified": False,
        "continuous_monitoring_verified": False,
        "market_news_coverage_verified": False,
    }
    if status_path.is_symlink() or not status_path.is_file():
        return report
    if status_path.stat().st_size > MAX_STATUS_BYTES:
        report["reason"] = "STATUS_FILE_TOO_LARGE"
        return report
    try:
        with status_path.open(encoding="utf-8") as stream:
            record = json.load(stream)
    except (OSError, ValueError, UnicodeError):
        report["reason"] = "INVALID_STATUS_FILE"
        return report
    if not isinstance(record, dict) or record.get("schema_version") != "integrated_news_pass_v1":
        report["reason"] = "NOT_AN_INTEGRATED_RUN"
        return report
    raw_time = record.get("finished_at")
    try:
        finished = datetime.fromisoformat(raw_time)
        if finished.utcoffset() is None:
            raise ValueError("Missing timestamp zone")
        age = (as_of.astimezone(UTC) - finished.astimezone(UTC)).total_seconds()
    except (TypeError, ValueError):
        report["reason"] = "INVALID_COMPLETION_TIME"
        return report
    if age < 0:
        report["reason"] = "FUTURE_COMPLETION_TIME"
        return report
    report["last_run_age_seconds"] = round(age, 3)
    if age > max_age_seconds:
        report["reason"] = "SCHEDULE_MISSED_OR_STALE"
        return report
    if record.get("status") != "healthy":
        report["reason"] = "LAST_RUN_NOT_HEALTHY"
        return report
    collection = record.get("collection")
    if not isinstance(collection, dict) or collection.get("status") != "healthy":
        report["reason"] = "COLLECTION_NOT_HEALTHY"
        return report
    outcomes = collection.get("source_results")
    if not isinstance(outcomes, dict) or set(outcomes) != set(expected_sources):
        report["reason"] = "SOURCE_SET_MISMATCH"
        return report
    if any(
        not isinstance(item, dict) or item.get("status") not in {"ok", "not_modified"}
        for item in outcomes.values()
    ):
        report["reason"] = "SOURCE_POLL_FAILED"
        return report
    if not isinstance(record.get("event_extraction"), dict):
        report["reason"] = "EVENT_EXTRACTION_NOT_RECORDED"
        return report
    report["status"] = "OBSERVED_RECENT_SUCCESS"
    report["reason"] = "RECENT_INTEGRATED_CYCLE"
    report["model_enabled_in_last_run"] = bool(
        isinstance(record.get("analysis"), dict)
        and record["analysis"].get("enabled") is True
    )
    report["new_articles_in_last_run"] = collection.get("new_articles")
    return report
