"""One read-only operational evidence packet from the EXISTING local news DB.

Never invents market coverage or treats a successful poll as headline recall.
No SQLite schema changes, API requests, model invocations or data exports.
"""
from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path

from tris_market_event_intelligence.event_lifecycle_latency import audit_event_lifecycle
from tris_market_event_intelligence.news_latency import audit_news_latency
from tris_market_event_intelligence.operations import OPERATIONAL_SOURCES
from tris_market_event_intelligence.session_operations import (
    SessionCalendar,
    market_session_plan,
)
from tris_market_event_intelligence.source_health import (
    assess_live_event,
    audit_source_health,
)

MAX_RECENT_ARTICLES = 50_000
EXPECTED_METRICS = ("news_latency", "event_lifecycle", "source_health", "recent_freshness")


def _recent_freshness(
    path: Path, *, now: datetime, since: datetime, sources: tuple[str, ...],
    freshness_seconds: int, maximum_rows: int = MAX_RECENT_ARTICLES,
) -> dict[str, object]:
    """Look only at first-seen rows in a bounded interval; no title/text reads."""
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        present = {r[0] for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        if "articles" not in present:
            raise ValueError("Articles table not present")
        count = db.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        if count > maximum_rows:
            raise ValueError("Article count exceeds bounded UTC-safe audit capacity")
        rows = db.execute(
            "SELECT source,detected_at,published_at FROM articles ORDER BY id",
        ).fetchall()
    # SQLite's lexical ISO timestamp ordering is NOT UTC ordering when rows
    # contain mixed offsets. Normalize in Python under the explicit row cap.
    recent: list[tuple[str, str, str | None]] = []
    for source, seen, published in rows:
        receipt = datetime.fromisoformat(seen)
        if receipt.utcoffset() is None:
            raise ValueError("First-seen timestamps must contain a timezone")
        if since <= receipt.astimezone(UTC) <= now:
            recent.append((source, seen, published))
    totals: dict[str, Counter[str]] = defaultdict(Counter)
    for source, seen, published in recent:
        reason = assess_live_event(
            detected_at=seen, published_at=published, now=now,
            max_received_age_seconds=freshness_seconds,
            max_declared_publish_age_seconds=freshness_seconds,
        )
        totals[source][reason] += 1
    return {
        "scope": "first_seen_in_observation_window",
        "since_utc": since.isoformat(),
        "until_utc": now.isoformat(),
        "freshness_budget_seconds": freshness_seconds,
        "total_articles": len(recent),
        "sources": [
            {
                "source": source,
                "articles": sum(totals[source].values()),
                "freshness_reasons": dict(sorted(totals[source].items())),
                "eligible_with_unverified_publication": totals[source][
                    "ELIGIBLE_WITH_UNVERIFIED_PUBLICATION"
                ],
            }
            for source in sorted(set(sources) | set(totals))
        ],
        "publisher_times_independently_verified": False,
        "news_completeness_measured": False,
    }


def build_operational_evidence(
    db_path: Path, *, as_of: datetime,
    sources: tuple[str, ...] = OPERATIONAL_SOURCES,
    window_hours: int = 24,
    freshness_minutes: int = 15,
    source_success_minutes: int = 15,
    calendar: SessionCalendar | None = None,
    markets: tuple[str, ...] = ("XKRX", "XNYS", "XNAS"),
) -> dict[str, object]:
    """Compose previous audits into a conservative, finite operational report.

    Each audit has its own failure reason. A missing/corrupt DB or legacy
    schema must NOT be presented as zero failures or zero latency.
    """
    if not isinstance(as_of, datetime) or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone aware")
    if not sources or len(set(sources)) != len(sources) or any(
        not isinstance(source, str) or not source for source in sources
    ):
        raise ValueError("sources must be distinct non-empty identifiers")
    if any(
        type(value) is not int or value < 1
        for value in (window_hours, freshness_minutes, source_success_minutes)
    ) or window_hours > 168 or freshness_minutes > 1440:
        raise ValueError("Invalid bounded audit configuration")
    now = as_of.astimezone(UTC)
    since = now - timedelta(hours=window_hours)
    plan = market_session_plan(calendar, now=now, markets=markets, sources=sources)

    checks: dict[str, dict[str, object]] = {}
    errors: list[str] = []
    db_exists = db_path.is_file() and not db_path.is_symlink()
    if db_exists:
        actions = {
            "news_latency": lambda: audit_news_latency(db_path, as_of=now),
            "event_lifecycle": lambda: audit_event_lifecycle(db_path, as_of=now),
            "recent_freshness": lambda: _recent_freshness(
                db_path, now=now, since=since, sources=sources,
                freshness_seconds=freshness_minutes * 60,
            ),
        }
        for name, action in actions.items():
            try:
                checks[name] = {"status": "OBSERVED", "data": action()}
            except (OSError, ValueError, TypeError, sqlite3.Error) as exc:
                checks[name] = {
                    "status": "UNAVAILABLE", "reason": type(exc).__name__,
                }
                errors.append(name)
    else:
        for name in ("news_latency", "event_lifecycle", "recent_freshness"):
            checks[name] = {"status": "UNAVAILABLE", "reason": "DATABASE_NOT_AVAILABLE"}
            errors.append(name)
    try:
        health = audit_source_health(
            db_path, expected_sources=sources, now=now,
            max_success_age_seconds=source_success_minutes * 60,
        )
        checks["source_health"] = {
            "status": "OBSERVED" if health["status"] == "OK" else "BLOCKED",
            "data": health,
        }
        if health["status"] != "OK":
            errors.append("source_health")
    except (OSError, ValueError, TypeError, sqlite3.Error) as exc:
        checks["source_health"] = {
            "status": "UNAVAILABLE", "reason": type(exc).__name__,
        }
        errors.append("source_health")

    acceptance = _evidence_acceptance(
        checks, requested_sources=sources,
        calendar_supplied=calendar is not None,
    )
    # Session-plan RUN means calendar *claims* the exchange is open; it is
    # not independently authenticated evidence of a live market.
    blockers = sorted(set(errors) | (
        {"exchange_sessions"} if plan["decision"] == "BLOCKED" else set()
    ))
    if not db_exists:
        status = "UNVERIFIED"
    elif blockers:
        status = "BLOCKED"
    else:
        status = "RESEARCH_OBSERVATIONS_ONLY"
    return {
        "schema_version": "operational_evidence_v1",
        "as_of_utc": now.isoformat(),
        "observation_start_utc": since.isoformat(),
        "status": status, "blocked_checks": blockers,
        "session_plan": plan, "checks": checks,
        "stage_1_evidence_review": acceptance,
        "sources_requested": list(sources),
        "real_market_data_observed_on_this_host": db_exists,
        "independent_publisher_first_seen_verified": False,
        "missing_headline_recall_measured": False,
        "gemma_first_usable_prediction_measured": False,
        "kr_us_market_session_accuracy_verified": False,
        "minute_price_feed_verified": False,
        "four_stage_product_complete": False,
        "note": (
            "Historical timing and local poll telemetry only. Only recent_freshness "
            "is window-scoped; legacy latency audits cover all bounded DB rows. "
            "Nothing measures unknown missing stories, true earliest publication, "
            "or actual intraday prediction quality."
        ),
    }


def _evidence_acceptance(
    checks: dict[str, dict[str, object]], *,
    requested_sources: tuple[str, ...],
    calendar_supplied: bool,
) -> dict[str, object]:
    """Identify missing observations without promoting local polling to live proof."""
    blockers: list[str] = []
    articles = checks.get("recent_freshness", {})
    if articles.get("status") != "OBSERVED":
        blockers.append("RECENT_NEWS_DATA_UNAVAILABLE")
    else:
        by_source = {
            row["source"]: row for row in articles["data"]["sources"]
        }
        for source in requested_sources:
            if by_source[source]["articles"] == 0:
                blockers.append("NO_RECENT_ARTICLES:" + source)
    event = checks.get("event_lifecycle", {})
    if event.get("status") != "OBSERVED":
        blockers.append("MODEL_TIMELINE_UNAVAILABLE")
    elif not event["data"]["model_checkpoints"]:
        blockers.append("NO_MODEL_CHECKPOINTS")
    if not calendar_supplied:
        blockers.append("REVIEWED_SESSION_CALENDAR_NOT_SUPPLIED")
    if checks.get("source_health", {}).get("status") != "OBSERVED":
        blockers.append("SOURCE_HEALTH_NOT_READY")
    blockers.extend((
        "INDEPENDENT_NEWS_COVERAGE_UNVERIFIED",
        "TRUE_PUBLISHER_BREAKING_TIME_UNVERIFIED",
        "KR_US_LIVE_MODEL_LATENCY_UNVERIFIED",
    ))
    return {
        "status": "NOT_VERIFIED",
        "blockers": sorted(set(blockers)),
        "reviewed_local_evidence_present": (
            articles.get("status") == "OBSERVED"
            and event.get("status") == "OBSERVED"
        ),
        "first_core_stage_complete": False,
        "note": (
            "Even a local run with articles and checkpoints cannot prove "
            "complete KR/US real-time market-news coverage."
        ),
    }
