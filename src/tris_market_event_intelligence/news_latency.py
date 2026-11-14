"""Read-only RSS first-seen latency and collection observability audit.

Source publication dates are provider claims, not authenticated news-break
timestamps. Response arrival is a *collection* observation, not proof of
first global publication. No writes or network requests.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

MAX_AUDIT_ROWS = 50_000


def _utc(value: str) -> datetime:
    stamp = datetime.fromisoformat(value)
    if stamp.utcoffset() is None:
        raise ValueError("Source timestamp must be timezone aware")
    return stamp.astimezone(UTC)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    return round(sorted_values[min(len(sorted_values) - 1, int(
        (len(sorted_values) - 1) * fraction + 0.999999999
    ))], 3)


def audit_news_latency(
    path: Path, *, as_of: datetime, maximum_rows: int = MAX_AUDIT_ROWS,
) -> dict[str, object]:
    """Inspect existing local news storage without creating DBs or tables."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("Existing non-symlink news DB required")
    if not 1 <= maximum_rows <= MAX_AUDIT_ROWS:
        raise ValueError("Audit row limit must be within 1..50000")
    if as_of.utcoffset() is None:
        raise ValueError("Audit as_of timestamp must be timezone-aware")
    now = as_of.astimezone(UTC)
    db_path = path.resolve()
    with closing(sqlite3.connect(
        f"{db_path.as_uri()}?mode=ro", uri=True,
    )) as db:
        db.execute("PRAGMA query_only=ON")
        tables = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        if not {"articles", "fetch_state", "captures"}.issubset(tables):
            raise ValueError("Existing news collector schema required")
        article_count = db.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        if article_count > maximum_rows:
            raise ValueError("Too many article rows for bounded audit")
        rows = db.execute(
            "SELECT source,published_at,detected_at FROM articles ORDER BY id"
        ).fetchall()
        state_rows = db.execute(
            "SELECT source,last_checked_at,last_success_at,last_error FROM fetch_state"
        ).fetchall()
        attempt_rows = []
        attempts_supported = "fetch_attempts" in tables
        if attempts_supported:
            attempts_count = db.execute(
                "SELECT COUNT(*) FROM fetch_attempts"
            ).fetchone()[0]
            if attempts_count > maximum_rows:
                raise ValueError("Too many fetch attempts for bounded audit")
            attempt_rows = db.execute(
                "SELECT source,request_started_at,response_at,completed_at,"
                "request_elapsed_ms,status,returned,inserted "
                "FROM fetch_attempts ORDER BY id"
            ).fetchall()
    groups: dict[str, dict[str, object]] = {}
    def item(source: str) -> dict[str, object]:
        return groups.setdefault(source, {
            "articles": 0, "published_timestamp_missing": 0,
            "published_timestamp_future_of_first_seen": 0,
            "valid_publish_to_first_seen_seconds": [],
            "request_elapsed_ms": [], "fetch_attempts": 0,
            "fetch_errors": 0, "fetch_not_modified": 0,
            "inserted": 0, "last_checked_at": None,
            "last_success_at": None, "last_error_recorded": False,
        })

    for source, publication, first_seen in rows:
        group = item(source)
        group["articles"] += 1
        detected = _utc(first_seen)
        if detected > now:
            raise ValueError("First-seen time after audit as_of")
        if publication is None:
            group["published_timestamp_missing"] += 1
            continue
        published = _utc(publication)
        lag = (detected - published).total_seconds()
        if lag < 0:
            group["published_timestamp_future_of_first_seen"] += 1
        else:
            group["valid_publish_to_first_seen_seconds"].append(lag)
    for source, checked, success, error in state_rows:
        group = item(source)
        group["last_checked_at"] = _utc(checked).isoformat()
        group["last_success_at"] = _utc(success).isoformat() if success else None
        group["last_error_recorded"] = error is not None
    for source, started, response, completed, elapsed, status, returned, inserted in attempt_rows:
        group = item(source)
        begin, end = _utc(started), _utc(completed)
        if begin > end or end > now:
            raise ValueError("Invalid collection attempt clock order")
        if response is not None and not begin <= _utc(response) <= end:
            raise ValueError("Invalid response arrival timestamp")
        if elapsed is not None:
            if not isinstance(elapsed, (int, float)) or not 0 <= elapsed <= 86400000:
                raise ValueError("Invalid request duration")
            group["request_elapsed_ms"].append(float(elapsed))
        if status not in {"ok", "not_modified", "error"}:
            raise ValueError("Unknown fetch outcome")
        if type(returned) is not int or type(inserted) is not int or not 0 <= inserted <= returned:
            raise ValueError("Invalid fetch article counts")
        group["fetch_attempts"] += 1
        group["inserted"] += inserted
        group["fetch_errors"] += status == "error"
        group["fetch_not_modified"] += status == "not_modified"
    sources: list[dict[str, object]] = []
    for source, group in sorted(groups.items()):
        lags = group.pop("valid_publish_to_first_seen_seconds")
        rtts = group.pop("request_elapsed_ms")
        checked = group["last_checked_at"]
        sources.append({
            "source": source, **group,
            "publish_to_first_seen_valid_count": len(lags),
            "publish_to_first_seen_median_seconds": round(median(lags), 3) if lags else None,
            "publish_to_first_seen_p95_seconds": _percentile(lags, 0.95),
            "fetch_request_p95_ms": _percentile(rtts, 0.95),
            "fetch_request_duration_count": len(rtts),
            "last_checked_age_seconds": (
                (now - _utc(checked)).total_seconds() if checked else None
            ),
        })
    return {
        "schema_version": "news_latency_audit_v1",
        "as_of_utc": now.isoformat(),
        "source_count": len(sources),
        "articles_count": article_count,
        "fetch_attempt_telemetry_available": attempts_supported,
        "sources": sources,
        "real_time_detection_verified": False,
        "publisher_timestamp_authenticated": False,
        "first_global_publication_verified": False,
        "note": (
            "First seen denotes local RSS receipt only. Publication comes from feed metadata; "
            "feeds may lag breaking news. Request duration is monotonic wall latency, "
            "not source publication lag. Real-time KR/US market coverage remains unverified."
        ),
    }
