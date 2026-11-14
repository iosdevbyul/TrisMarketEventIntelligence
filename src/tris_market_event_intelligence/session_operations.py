"""Explicit KR/US market-session operating plan and safe source readiness.

Offline inspection only. Open/close dates come from supplied reviewed CSV,
never from guessed weekdays. A feed can matter cross-border regardless of origin.
"""
from __future__ import annotations

import csv
from datetime import UTC, date, datetime
from pathlib import Path

from tris_market_event_intelligence.exchange_sessions import (
    MARKET_ZONES,
    SessionCalendar,
    SessionDay,
    utc,
)
from tris_market_event_intelligence.operations import OPERATIONAL_SOURCES
from tris_market_event_intelligence.source_health import audit_source_health

MAX_CALENDAR_BYTES = 1_048_576
MAX_SESSION_ROWS = 2000
SUPPORTED_MARKETS = ("XKRX", "XNYS", "XNAS")


def load_session_calendar(path: Path) -> SessionCalendar:
    """Load operator-supplied exact session dates, reject unknown/huge sources."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("Reviewed calendar must be an existing non-symlink CSV")
    if path.stat().st_size > MAX_CALENDAR_BYTES:
        raise ValueError("Calendar CSV too large")
    days: list[SessionDay] = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        if rows.fieldnames != ["market", "date", "status", "open_at", "close_at"]:
            raise ValueError("Calendar requires exact market/date/status/open_at/close_at columns")
        for index, row in enumerate(rows, 1):
            if index > MAX_SESSION_ROWS:
                raise ValueError("Calendar rows exceed cap")
            if any(value is None for value in row.values()) or None in row:
                raise ValueError("Malformed calendar row")
            days.append(SessionDay(
                market=row["market"].strip(),
                day=date.fromisoformat(row["date"].strip()),
                status=row["status"].strip(),
                open_at=(
                    datetime.fromisoformat(row["open_at"].strip())
                    if row["open_at"].strip() else None
                ),
                close_at=(
                    datetime.fromisoformat(row["close_at"].strip())
                    if row["close_at"].strip() else None
                ),
            ))
    return SessionCalendar(days)


def market_session_plan(
    calendar: SessionCalendar | None, *,
    now: datetime, markets: tuple[str, ...] = SUPPORTED_MARKETS,
    sources: tuple[str, ...] = OPERATIONAL_SOURCES,
) -> dict[str, object]:
    """Operational gate for market-focused processing, NOT global news collecting.

    Unknown dates fail closed, even when another market is open.
    Session closure only means *market-focused cycle* skipped, not that
    important after-hours news should be discarded by other collectors.
    """
    instant = utc(now)
    if not markets or len(markets) != len(set(markets)) or (
        set(markets) - set(SUPPORTED_MARKETS)
    ):
        raise ValueError("Markets must be unique supported exchange identifiers")
    if not sources or len(sources) != len(set(sources)) or (
        set(sources) - set(OPERATIONAL_SOURCES)
    ):
        raise ValueError("Sources must be unique approved RSS identifiers")
    status = {
        market: calendar.instant_status(market, instant) if calendar is not None
        else "CALENDAR_UNAVAILABLE"
        for market in markets
    }
    if "CALENDAR_UNAVAILABLE" in status.values():
        decision, reason = "BLOCKED", "EXPLICIT_CALENDAR_REQUIRED"
    elif "IN_SESSION" in status.values():
        decision, reason = "RUN", "AT_LEAST_ONE_MARKET_OPEN"
    else:
        decision, reason = "SKIP", "NO_REQUESTED_MARKET_IN_SESSION"
    return {
        "schema_version": "market_session_plan_v1",
        "evaluated_at_utc": instant.isoformat(),
        "markets": status, "sources": list(sources),
        "decision": decision, "reason": reason,
        "calendar_independently_verified": False,
        "live_market_coverage_verified": False,
        "source_geography_is_a_filter": False,
        "note": (
            "All configured RSS feeds are preserved when running: macroeconomic "
            "news can affect either market. Unknown sessions block, closed "
            "sessions skip only this market-focused cycle, not the separate "
            "round-the-clock news discovery. Calendar content is operator-supplied."
        ),
    }


def operational_readiness(
    database: Path, *, calendar: SessionCalendar | None,
    now: datetime, markets: tuple[str, ...] = SUPPORTED_MARKETS,
    sources: tuple[str, ...] = OPERATIONAL_SOURCES,
    max_success_age_seconds: int = 900,
) -> dict[str, object]:
    """Combine session routing with actual persisted source polling evidence."""
    plan = market_session_plan(calendar, now=now, markets=markets, sources=sources)
    health = audit_source_health(
        database, now=now, expected_sources=sources,
        max_success_age_seconds=max_success_age_seconds,
    )
    decision = plan["decision"]
    if decision == "RUN" and health["status"] != "OK":
        decision = "BLOCKED"
    return {
        "schema_version": "market_operations_readiness_v1",
        "decision": decision,
        "session_plan": plan,
        "source_health": health,
        "kr_us_realtime_verified": False,
        "local_gemma_runtime_verified": False,
        "minute_prices_available_verified": False,
        "ready_to_train_or_predict": False,
    }
