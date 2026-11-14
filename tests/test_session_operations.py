"""Explicit KR/US calendar and no-network operating gate regressions."""
from __future__ import annotations

import csv
import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import httpx
import pytest

from tris_market_event_intelligence.collector import NewsStore
from tris_market_event_intelligence.exchange_sessions import SessionCalendar, SessionDay
from tris_market_event_intelligence.integrated_pipeline import run_integrated_pass
from tris_market_event_intelligence.integrated_pipeline_cli import main as integrated_main
from tris_market_event_intelligence.session_operations import (
    load_session_calendar,
    market_session_plan,
    operational_readiness,
)
from tris_market_event_intelligence.session_operations_cli import main as readiness_main

FIXED = datetime(2026, 11, 16, 15, 0, tzinfo=UTC)


def ny_open() -> SessionDay:
    return SessionDay(
        "XNAS", date(2026, 11, 16), "OPEN",
        datetime(2026, 11, 16, 14, 30, tzinfo=UTC),
        datetime(2026, 11, 16, 21, 0, tzinfo=UTC),
    )


def fixture_calendar(tmp_path: Path, rows: list[SessionDay]) -> Path:
    path = tmp_path / "reviewed.csv"
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["market", "date", "status", "open_at", "close_at"])
        for day in rows:
            writer.writerow([
                day.market, day.day.isoformat(), day.status,
                day.open_at.isoformat() if day.open_at else "",
                day.close_at.isoformat() if day.close_at else "",
            ])
    return path


def test_us_open_kr_closed_no_duplicate_or_fake_schedule(tmp_path: Path) -> None:
    path = fixture_calendar(tmp_path, [
        ny_open(), SessionDay("XKRX", date(2026, 11, 17), "CLOSED"),
    ])
    plan = market_session_plan(
        load_session_calendar(path), now=FIXED,
        markets=("XKRX", "XNAS"), sources=("fed_monetary", "bok_all"),
    )
    assert plan["decision"] == "RUN"
    assert plan["markets"] == {"XKRX": "MARKET_CLOSED", "XNAS": "IN_SESSION"}
    assert plan["sources"] == ["fed_monetary", "bok_all"]
    assert plan["source_geography_is_a_filter"] is False
    assert plan["calendar_independently_verified"] is False


def test_both_closed_or_before_open_skip_without_24h_collector_claim() -> None:
    closed = SessionCalendar([
        SessionDay("XNAS", date(2026, 11, 16), "CLOSED"),
        SessionDay("XKRX", date(2026, 11, 17), "CLOSED"),
    ])
    result = market_session_plan(
        closed, now=FIXED, markets=("XKRX", "XNAS"),
    )
    assert result["decision"] == "SKIP"
    assert result["reason"] == "NO_REQUESTED_MARKET_IN_SESSION"
    before = market_session_plan(
        SessionCalendar([ny_open()]),
        now=datetime(2026, 11, 16, 14, 0, tzinfo=UTC),
        markets=("XNAS",),
    )
    assert before["decision"] == "SKIP"
    assert before["markets"]["XNAS"] == "OUTSIDE_REGULAR_SESSION"


def test_absent_holiday_or_extra_market_fails_closed() -> None:
    plan = market_session_plan(
        SessionCalendar([ny_open()]), now=FIXED,
        markets=("XNYS", "XNAS"),
    )
    assert plan["decision"] == "BLOCKED"
    assert plan["markets"]["XNYS"] == "CALENDAR_UNAVAILABLE"
    assert market_session_plan(
        None, now=FIXED, markets=("XNAS",),
    )["reason"] == "EXPLICIT_CALENDAR_REQUIRED"


def test_us_dst_and_operator_declared_early_close() -> None:
    july = SessionCalendar([
        SessionDay(
            "XNYS", date(2026, 7, 2), "OPEN",
            datetime(2026, 7, 2, 9, 30, tzinfo=ZoneInfo("America/New_York")),
            datetime(2026, 7, 2, 13, 0, tzinfo=ZoneInfo("America/New_York")),
        ),
    ])
    assert market_session_plan(
        july, now=datetime(2026, 7, 2, 16, tzinfo=UTC),
        markets=("XNYS",),
    )["decision"] == "RUN"
    assert market_session_plan(
        july, now=datetime(2026, 7, 2, 18, tzinfo=UTC),
        markets=("XNYS",),
    )["decision"] == "SKIP"


def test_explicit_csv_rejects_duplicate_malformed_or_symlink(tmp_path: Path) -> None:
    item = ny_open()
    path = fixture_calendar(tmp_path, [item, item])
    with pytest.raises(ValueError, match="Duplicate"):
        load_session_calendar(path)
    path = fixture_calendar(tmp_path, [item])
    data = path.read_text().replace("market,date", "bad,date")
    path.write_text(data)
    with pytest.raises(ValueError, match="exact"):
        load_session_calendar(path)
    target = tmp_path / "alias.csv"
    target.symlink_to(path)
    with pytest.raises(ValueError, match="non-symlink"):
        load_session_calendar(target)


def test_readiness_combines_session_and_real_local_poll_evidence(tmp_path: Path) -> None:
    path = tmp_path / "news.sqlite3"
    store = NewsStore(path)
    store.initialize()
    stamp = FIXED.isoformat()
    store.save("fed_monetary", "https://example.org", None, [], stamp, None, None)
    store.record_attempt("fed_monetary", stamp, stamp, stamp, 1.0, "ok", 0, 0)
    calendar = SessionCalendar([ny_open()])
    report = operational_readiness(
        path, calendar=calendar, now=FIXED,
        markets=("XNAS",), sources=("fed_monetary",),
    )
    assert report["decision"] == "RUN"
    assert report["source_health"]["status"] == "OK"
    assert report["kr_us_realtime_verified"] is False
    assert report["ready_to_train_or_predict"] is False
    bad = operational_readiness(
        path, calendar=calendar, now=FIXED,
        markets=("XNAS",), sources=("fed_monetary", "bok_all"),
    )
    assert bad["decision"] == "BLOCKED"
    assert bad["source_health"]["status"] == "BLOCKED"
    stale = operational_readiness(
        path, calendar=calendar, now=FIXED + timedelta(minutes=30),
        markets=("XNAS",), sources=("fed_monetary",),
    )
    assert stale["decision"] == "BLOCKED"


def test_session_skipped_pipeline_does_not_create_db_or_call_feed(tmp_path: Path) -> None:
    path = tmp_path / "not-created" / "news.sqlite3"
    clock = datetime.now(UTC)
    local_date = clock.astimezone(ZoneInfo("America/New_York")).date()
    calendar = SessionCalendar([SessionDay("XNAS", local_date, "CLOSED")])
    with httpx.Client(transport=httpx.MockTransport(
        lambda _: pytest.fail("Closed market cannot fetch"),
    )) as client:
        output = run_integrated_pass(
            path, selected_sources=("fed_monetary",), client=client,
            session_calendar=calendar, target_markets=("XNAS",),
        )
    assert output["status"] == "session_skipped"
    assert output["network_requests"] == 0
    assert output["model_invoked"] is False
    assert not path.exists() and not path.parent.exists()


def test_unknown_session_pipeline_blocks_before_network_or_model(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite3"
    with httpx.Client(transport=httpx.MockTransport(
        lambda _: pytest.fail("Unknown market schedule cannot fetch"),
    )) as client:
        output = run_integrated_pass(
            path, selected_sources=("fed_monetary",), client=client,
            session_calendar=SessionCalendar([ny_open()]),
            target_markets=("XNYS",),
        )
    assert output["status"] == "blocked"
    assert output["session_plan"]["reason"] == "EXPLICIT_CALENDAR_REQUIRED"
    assert not path.exists()


def test_integrated_cli_closed_session_avoids_even_local_model_probe(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    now = datetime.now(UTC)
    local_date = now.astimezone(ZoneInfo("America/New_York")).date()
    sessions = fixture_calendar(tmp_path, [SessionDay("XNAS", local_date, "CLOSED")])
    path = tmp_path / "no-network.sqlite3"
    with (
        patch.object(sys, "argv", [
            "market-event-integrated", "--sessions", str(sessions),
            "--markets", "XNAS", "--sources", "fed_monetary",
            "--db", str(path), "--enable-local-analysis",
        ]),
        patch(
            "tris_market_event_intelligence.integrated_pipeline_cli.httpx.get",
            side_effect=AssertionError("Ollama must not be probed"),
        ),
    ):
        integrated_main()
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "session_skipped"
    assert not path.exists()


def test_readiness_cli_returns_blocked_without_db_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    path = fixture_calendar(tmp_path, [ny_open()])
    missing = tmp_path / "missing.sqlite3"
    with (
        patch.object(sys, "argv", [
            "market-event-ops-readiness", "--sessions", str(path),
            "--db", str(missing), "--markets", "XNAS",
            "--sources", "fed_monetary",
            "--as-of", FIXED.isoformat(),
        ]),
        pytest.raises(SystemExit) as raised,
    ):
        readiness_main()
    assert raised.value.code == 2
    result = json.loads(capsys.readouterr().out)
    assert result["decision"] == "BLOCKED"
    assert result["source_health"]["reason"] == "SOURCE_DB_UNAVAILABLE"
    assert not missing.exists()
