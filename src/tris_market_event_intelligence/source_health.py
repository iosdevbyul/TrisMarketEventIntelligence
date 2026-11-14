"""Point-in-time operational safeguards for news sources and live analysis.

Uses only existing local SQLite observations and source-declared publication
metadata. Source health and freshness DO NOT establish global news coverage.
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path


def _instant(value: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError("Expected timestamp text")
    value_at = datetime.fromisoformat(value)
    if value_at.tzinfo is None or value_at.utcoffset() is None:
        raise ValueError("Timestamp must carry a timezone")
    return value_at.astimezone(UTC)


def assess_live_event(
    *,
    detected_at: str,
    published_at: str | None,
    now: datetime,
    max_received_age_seconds: int,
    max_declared_publish_age_seconds: int,
) -> str:
    """Return a stable reason code; never turn unverified dates into facts."""
    if now.utcoffset() is None:
        raise ValueError("now must be timezone aware")
    if min(max_received_age_seconds, max_declared_publish_age_seconds) < 1:
        raise ValueError("Freshness budgets must be positive")
    clock = now.astimezone(UTC)
    seen = _instant(detected_at)
    if seen > clock:
        return "RECEIVED_IN_FUTURE"
    if (clock - seen).total_seconds() > max_received_age_seconds:
        return "RECEIPT_STALE"
    if published_at is None:
        return "PUBLICATION_UNVERIFIED"
    published = _instant(published_at)
    if published > seen:
        return "PUBLICATION_AFTER_RECEIPT"
    if (seen - published).total_seconds() > max_declared_publish_age_seconds:
        return "DECLARED_PUBLICATION_STALE"
    return "ELIGIBLE_WITH_UNVERIFIED_PUBLICATION"


def audit_source_health(
    database: Path,
    *,
    expected_sources: tuple[str, ...],
    now: datetime,
    max_success_age_seconds: int = 900,
    maximum_consecutive_failures: int = 3,
) -> dict[str, object]:
    """Offline no-write source telemetry; a recent 304 counts as successful polling."""
    if not expected_sources or len(set(expected_sources)) != len(expected_sources):
        raise ValueError("Expected unique explicitly configured sources")
    if now.utcoffset() is None:
        raise ValueError("now must be timezone aware")
    if max_success_age_seconds < 1 or maximum_consecutive_failures < 1:
        raise ValueError("Health thresholds must be positive")
    clock = now.astimezone(UTC)
    if database.is_symlink() or not database.is_file():
        return {
            "status": "BLOCKED",
            "reason": "SOURCE_DB_UNAVAILABLE",
            "sources": [],
            "monitoring_verified": False,
            "live_news_coverage_verified": False,
        }
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        tables = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        if not {"fetch_attempts", "fetch_state"}.issubset(tables):
            return {
                "status": "BLOCKED",
                "reason": "SOURCE_TELEMETRY_UNAVAILABLE",
                "sources": [],
                "monitoring_verified": False,
                "live_news_coverage_verified": False,
            }
        # Fetch recent attempts only, bounded by source; check state for exact
        # last successful observation. No full scans of historic article text.
        rows = db.execute(
            "SELECT source,status,completed_at FROM fetch_attempts "
            "ORDER BY id DESC LIMIT 5000"
        ).fetchall()
        states = db.execute(
            "SELECT source,last_checked_at,last_success_at,last_error FROM fetch_state"
        ).fetchall()
    by_source: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for source, status, complete in rows:
        if status not in {"ok", "not_modified", "error"}:
            raise ValueError("Invalid persisted request outcome")
        by_source[source].append((status, complete))
    state_map = {row[0]: row[1:] for row in states}
    summaries = []
    for source in expected_sources:
        status_rows = by_source.get(source, [])
        streak = 0
        for outcome, _ in status_rows:
            if outcome != "error":
                break
            streak += 1
        checkpoint = state_map.get(source)
        success_at = _instant(checkpoint[1]) if checkpoint and checkpoint[1] else None
        checked_at = _instant(checkpoint[0]) if checkpoint and checkpoint[0] else None
        if success_at is not None and success_at > clock:
            reason = "FUTURE_SOURCE_SUCCESS"
        elif checked_at is not None and checked_at > clock:
            reason = "FUTURE_SOURCE_CHECK"
        elif success_at is None:
            reason = "NO_SUCCESSFUL_POLL"
        elif not status_rows:
            reason = "NO_REQUEST_TELEMETRY"
        elif (clock - success_at).total_seconds() > max_success_age_seconds:
            reason = "SOURCE_POLL_STALE"
        elif streak >= maximum_consecutive_failures:
            reason = "CONSECUTIVE_FAILURES"
        elif checkpoint and checkpoint[2] == "HTTP_429":
            reason = "SOURCE_RATE_LIMITED"
        elif checkpoint and checkpoint[2] in {"HTTP_401", "HTTP_403"}:
            reason = "SOURCE_ACCESS_DENIED"
        elif streak > 0:
            reason = "SOURCE_RECENT_ERROR"
        else:
            reason = "RECENT_SUCCESSFUL_POLL"
        summaries.append({
            "source": source,
            "status": "OK" if reason == "RECENT_SUCCESSFUL_POLL" else "BLOCKED",
            "reason": reason,
            "consecutive_failed_attempts": streak,
            "last_success_age_seconds": (
                round((clock - success_at).total_seconds(), 3) if success_at else None
            ),
            "last_checked_age_seconds": (
                round((clock - checked_at).total_seconds(), 3) if checked_at else None
            ),
            "recent_attempts_inspected": len(status_rows),
        })
    return {
        "schema_version": "source_health_v1",
        "as_of_utc": clock.isoformat(),
        "status": "OK" if all(s["status"] == "OK" for s in summaries) else "BLOCKED",
        "max_success_age_seconds": max_success_age_seconds,
        "maximum_consecutive_failures": maximum_consecutive_failures,
        "sources": summaries,
        "monitoring_verified": False,
        "live_news_coverage_verified": False,
        "note": (
            "Recent successful polling measures collection reachability, NOT news coverage. "
            "No historical source freshness or publisher time is invented."
        ),
    }
