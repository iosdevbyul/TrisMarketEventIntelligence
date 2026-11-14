"""Fail-closed KR/US exchange-day, early-close and trading-halt regressions."""
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tris_market_event_intelligence.exchange_sessions import (
    SessionCalendar,
    SessionDay,
    TradingHalt,
)
from tris_market_event_intelligence.intraday_event_study import (
    IntradayNews,
    MinuteClose,
    study_intraday,
)
from tris_market_event_intelligence.intraday_event_study_cli import intraday_csv


def stamp(market: str, day: int, hour: int, minute: int) -> datetime:
    zone = ZoneInfo("Asia/Seoul" if market == "XKRX" else "America/New_York")
    return datetime(2026, 11, day, hour, minute, tzinfo=zone)


def session(
    market: str = "XNYS", day: int = 27, *, close_hour: int = 13,
) -> SessionDay:
    opening = stamp(market, day, 9, 30) if market != "XKRX" else stamp(
        market, day, 9, 0,
    )
    return SessionDay(
        market, date(2026, 11, day), "OPEN",
        opening, stamp(market, day, close_hour, 0),
    )


def news(market: str = "XNYS", day: int = 27, hour: int = 12, minute: int = 55) -> IntradayNews:
    publication = stamp(market, day, hour, minute)
    return IntradayNews(
        1, market, "NVDA" if market != "XKRX" else "005930", "SPY",
        publication, publication, publication,
    )


def minutes(
    market: str = "XNYS", day: int = 27, hour: int = 12, minute: int = 56,
    count: int = 5,
) -> list[MinuteClose]:
    opening = stamp(market, day, hour, minute)
    return [
        MinuteClose(market, symbol, opening + timedelta(minutes=i), 100.0 + i)
        for symbol in ("NVDA" if market != "XKRX" else "005930", "SPY")
        for i in range(count)
    ]


def test_explicit_closed_us_day_is_not_turned_into_training_example() -> None:
    calendar = SessionCalendar([SessionDay("XNYS", date(2026, 11, 26), "CLOSED")])
    row = study_intraday(
        [news(day=26, hour=10, minute=15)], [], calendar=calendar,
    )[0]
    assert row.status == "MARKET_CLOSED"
    assert all(label.excess_return is None for label in row.labels)


def test_unlisted_market_date_is_unknown_not_assumed_closed() -> None:
    calendar = SessionCalendar([session()])
    row = study_intraday(
        [news(day=25, hour=10, minute=15)], [], calendar=calendar,
    )[0]
    assert row.status == "CALENDAR_UNAVAILABLE"
    assert row.latency_seconds == 0


def test_early_close_blocks_later_labels_and_excludes_postclose_bars() -> None:
    calendar = SessionCalendar([session(close_hour=13)])
    row = study_intraday(
        [news()], minutes(), calendar=calendar,
    )[0]
    assert row.entry_at == stamp("XNYS", 27, 12, 56).astimezone(UTC).isoformat()
    assert row.labels[0].status == "COMPLETED"
    assert row.labels[1].status == "SESSION_BOUNDARY"
    assert row.labels[4].status == "SESSION_BOUNDARY"
    with pytest.raises(ValueError, match="confirmed exchange session"):
        study_intraday([], [
            MinuteClose("XNYS", "NVDA", stamp("XNYS", 27, 13, 1), 100.0),
        ], calendar=calendar)


def test_trading_halt_blocks_label_even_if_vendor_has_forward_filled_bars() -> None:
    halt = TradingHalt(
        "XNYS", "NVDA", stamp("XNYS", 27, 12, 58),
        stamp("XNYS", 27, 12, 59),
    )
    calendar = SessionCalendar([session(close_hour=16)], [halt])
    row = study_intraday([news()], minutes(), calendar=calendar)[0]
    assert row.labels[0].status == "COMPLETED"
    assert row.labels[1].status == "TRADING_HALTED_WINDOW"
    assert row.labels[1].excess_return is None


def test_halt_at_entry_blocks_every_horizon() -> None:
    calendar = SessionCalendar([
        session(close_hour=16),
    ], [TradingHalt(
        "XNYS", "NVDA", stamp("XNYS", 27, 12, 56), stamp("XNYS", 27, 12, 58),
    )])
    row = study_intraday([news()], minutes(), calendar=calendar)[0]
    assert row.status == "TRADING_HALTED_AT_ENTRY"
    assert all(label.excess_return is None for label in row.labels)


def test_halt_on_benchmark_also_blocks_labels() -> None:
    calendar = SessionCalendar([
        session(close_hour=16),
    ], [TradingHalt(
        "XNYS", "SPY", stamp("XNYS", 27, 12, 58), stamp("XNYS", 27, 12, 59),
    )])
    row = study_intraday([news()], minutes(), calendar=calendar)[0]
    assert row.labels[1].status == "TRADING_HALTED_WINDOW"


def test_session_validation_rejects_missing_invalid_and_duplicate_days() -> None:
    with pytest.raises(ValueError, match="at least one"):
        SessionCalendar([])
    with pytest.raises(ValueError, match="Duplicate market"):
        SessionCalendar([session(), session()])
    with pytest.raises(ValueError, match="Closed session"):
        SessionDay(
            "XNYS", date(2026, 11, 27), "CLOSED",
            stamp("XNYS", 27, 9, 30), None,
        )
    with pytest.raises(ValueError, match="explicit timezone"):
        SessionDay(
            "XNYS", date(2026, 11, 27), "OPEN",
            datetime(2026, 11, 27, 9, 30), stamp("XNYS", 27, 13, 0),
        )
    with pytest.raises(ValueError, match="exchange-local"):
        SessionDay(
            "XNYS", date(2026, 11, 26), "OPEN",
            stamp("XNYS", 27, 9, 30), stamp("XNYS", 27, 13, 0),
        )


def test_halt_overlap_and_invalid_timestamps_rejected() -> None:
    with pytest.raises(ValueError, match="Halt start"):
        TradingHalt(
            "XNYS", "NVDA", stamp("XNYS", 27, 12, 59), stamp("XNYS", 27, 12, 58),
        )
    a = TradingHalt(
        "XNYS", "NVDA", stamp("XNYS", 27, 12, 56), stamp("XNYS", 27, 12, 59),
    )
    b = TradingHalt(
        "XNYS", "NVDA", stamp("XNYS", 27, 12, 58), stamp("XNYS", 27, 13, 0),
    )
    with pytest.raises(ValueError, match="Overlapping"):
        SessionCalendar([session()], [a, b])


def test_calendar_csv_provides_explicit_closed_day(tmp_path: Path) -> None:
    events = tmp_path / "events.csv"
    bars = tmp_path / "prices.csv"
    schedules = tmp_path / "schedule.csv"
    events.write_text(
        "event_id,market,symbol,benchmark_symbol,published_at,received_at,available_at\n"
        "1,XNAS,NVDA,SPY,2026-11-26T10:15:00-05:00,"
        "2026-11-26T10:15:00-05:00,2026-11-26T10:15:00-05:00\n",
        encoding="utf-8",
    )
    bars.write_text(
        "market,symbol,close_at,adjusted_close,adjustment\n",
        encoding="utf-8",
    )
    schedules.write_text(
        "market,date,status,open_at,close_at\n"
        "XNAS,2026-11-26,CLOSED,,\n",
        encoding="utf-8",
    )
    originals = events.read_bytes(), bars.read_bytes(), schedules.read_bytes()
    result = intraday_csv(events, bars, sessions_path=schedules)
    assert result["session_calendar_supplied"] is True
    assert result["session_calendar_source_verified"] is False
    assert result["items"][0]["status"] == "MARKET_CLOSED"
    assert result["completed_label_count"] == 0
    assert originals == (events.read_bytes(), bars.read_bytes(), schedules.read_bytes())


def test_csv_halts_requires_session_schedule(tmp_path: Path) -> None:
    events, bars, halts = [tmp_path / name for name in (
        "events.csv", "bars.csv", "halts.csv",
    )]
    events.write_text(
        "event_id,market,symbol,benchmark_symbol,published_at,received_at,available_at\n",
        encoding="utf-8",
    )
    bars.write_text(
        "market,symbol,close_at,adjusted_close,adjustment\n",
        encoding="utf-8",
    )
    halts.write_text(
        "market,symbol,start_at,end_at\n", encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires"):
        intraday_csv(events, bars, halts_path=halts)


def test_us_dst_respected_by_calendar_local_date() -> None:
    schedule = SessionDay(
        "XNYS", date(2026, 7, 6), "OPEN",
        datetime(2026, 7, 6, 9, 30, tzinfo=ZoneInfo("America/New_York")),
        datetime(2026, 7, 6, 16, 0, tzinfo=ZoneInfo("America/New_York")),
    )
    calendar = SessionCalendar([schedule])
    assert calendar.instant_status(
        "XNYS", datetime(2026, 7, 6, 13, 30, tzinfo=UTC), include_open=True,
    ) == "IN_SESSION"
