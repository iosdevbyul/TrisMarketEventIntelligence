"""Cross-market news timing and strict minute-label integrity regressions."""
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tris_market_event_intelligence.intraday_event_study import (
    IntradayNews,
    MinuteClose,
    study_intraday,
)
from tris_market_event_intelligence.intraday_event_study_cli import intraday_csv


def instant(market: str, hour: int, minute: int) -> datetime:
    zone = ZoneInfo("Asia/Seoul" if market == "XKRX" else "America/New_York")
    return datetime(2026, 11, 16, hour, minute, tzinfo=zone)


def event(market: str = "XKRX", *, available_offset: int = 0) -> IntradayNews:
    stamp = instant(market, 10, 15)
    return IntradayNews(
        7, market, "005930" if market == "XKRX" else "NVDA",
        "KOSPI_IDX" if market == "XKRX" else "SPY",
        stamp, stamp + timedelta(seconds=45),
        stamp + timedelta(seconds=45 + available_offset),
    )


def minute_series(market: str = "XKRX", count: int = 63) -> list[MinuteClose]:
    origin = instant(market, 10, 16)
    e = event(market)
    return [
        MinuteClose(
            market, sym, origin + timedelta(minutes=i),
            price + i * step,
        )
        for sym, price, step in [
            (e.symbol, 100.0, 1.0),
            (e.benchmark_symbol, 100.0, 0.2),
        ]
        for i in range(count)
    ]


@pytest.mark.parametrize("market", ["XKRX", "XNYS", "XNAS"])
def test_regular_session_news_uses_first_fully_closed_minute_after_availability(
    market: str,
) -> None:
    row = study_intraday([event(market)], minute_series(market))[0]
    assert row.status == "COMPLETED"
    assert row.entry_at == instant(market, 10, 16).astimezone(UTC).isoformat()
    assert row.latency_seconds == 45
    assert [label.horizon_minutes for label in row.labels] == [1, 5, 15, 30, 60]
    assert row.labels[0].stock_return == pytest.approx(0.01)
    assert row.labels[0].benchmark_return == pytest.approx(0.002)
    assert row.labels[0].excess_return == pytest.approx(0.008)
    assert row.labels[-1].exit_at == instant(market, 11, 16).astimezone(UTC).isoformat()


def test_delayed_ingestion_must_not_reuse_past_minute() -> None:
    observed = event(available_offset=600)
    row = study_intraday([observed], minute_series())[0]
    assert row.entry_at == instant("XKRX", 10, 26).astimezone(UTC).isoformat()
    assert row.latency_seconds == 645
    assert row.labels[0].stock_return == pytest.approx(1 / 110)
    assert row.labels[0].benchmark_return == pytest.approx(0.2 / 102)


def test_missing_minute_blocks_label_instead_of_compressing_horizon() -> None:
    bars = [
        bar for bar in minute_series()
        if not (bar.symbol == "005930" and bar.close_at.minute == 19)
    ]
    row = study_intraday([event()], bars)[0]
    assert row.labels[0].status == "COMPLETED"
    assert row.labels[1].status == "MISSING_MINUTE_BAR"
    assert row.labels[1].excess_return is None
    assert row.labels[2].status == "MISSING_MINUTE_BAR"


def test_market_closing_boundary_is_not_crossed() -> None:
    e = IntradayNews(
        8, "XNYS", "NVDA", "SPY",
        instant("XNYS", 15, 55),
        instant("XNYS", 15, 55),
        instant("XNYS", 15, 55),
    )
    bars = [
        MinuteClose("XNYS", sym, instant("XNYS", 15, 56) + timedelta(minutes=i), 100 + i)
        for sym in ("NVDA", "SPY") for i in range(5)
    ]
    row = study_intraday([e], bars)[0]
    assert row.labels[0].status == "COMPLETED"
    assert row.labels[5 - 1].status == "SESSION_BOUNDARY"
    assert row.labels[-1].status == "SESSION_BOUNDARY"


@pytest.mark.parametrize("market,hour", [("XKRX", 8), ("XNYS", 8), ("XNAS", 17)])
def test_pre_market_and_after_hours_not_treated_as_regular(market: str, hour: int) -> None:
    stamp = instant(market, hour, 10)
    e = IntradayNews(1, market, "STOCK", "MARKET", stamp, stamp, stamp)
    row = study_intraday([e], [])[0]
    assert row.status == "OUTSIDE_REGULAR_SESSION"
    assert all(label.excess_return is None for label in row.labels)


def test_available_at_close_has_no_entry_in_session() -> None:
    stamp = instant("XKRX", 15, 30)
    row = study_intraday([
        IntradayNews(4, "XKRX", "005930", "KOSPI_IDX", stamp, stamp, stamp),
    ], [])[0]
    assert row.status == "NO_ENTRY_IN_SESSION"


def test_news_timestamp_chronology_and_timezone_must_be_valid() -> None:
    valid = event()
    with pytest.raises(ValueError, match="chronological"):
        study_intraday([
            IntradayNews(3, valid.market, valid.symbol, valid.benchmark_symbol,
                         valid.received_at, valid.published_at, valid.available_at),
        ], [])
    with pytest.raises(ValueError, match="timezone"):
        study_intraday([
            IntradayNews(3, valid.market, valid.symbol, valid.benchmark_symbol,
                         valid.published_at.replace(tzinfo=None),
                         valid.received_at, valid.available_at),
        ], [])


def test_missing_benchmark_entry_prevents_returns() -> None:
    rows = [r for r in minute_series() if not (
        r.symbol == "KOSPI_IDX" and r.close_at == instant("XKRX", 10, 16)
    )]
    result = study_intraday([event()], rows)[0]
    assert result.status == "BENCHMARK_ENTRY_MISSING"
    assert all(label.excess_return is None for label in result.labels)


@pytest.mark.parametrize("invalid", [0, -1, float("inf"), float("nan")])
def test_invalid_minute_prices_are_rejected(invalid: float) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        study_intraday([], [
            MinuteClose("XKRX", "005930", instant("XKRX", 10, 16), invalid),
        ])


def test_duplicate_bars_duplicate_event_and_invalid_market_rejected() -> None:
    bars = minute_series()
    with pytest.raises(ValueError, match="Duplicate minute"):
        study_intraday([], bars + [bars[0]])
    with pytest.raises(ValueError, match="Duplicate event"):
        study_intraday([event(), event()], bars)
    with pytest.raises(ValueError, match="supported market"):
        study_intraday([
            IntradayNews(8, "INVALID", "A", "B",
                         event().published_at, event().received_at, event().available_at),
        ], [])


def test_utc_observed_event_and_dst_timezone_for_us() -> None:
    e = event("XNYS")
    utc_event = IntradayNews(
        e.event_id, e.market, e.symbol, e.benchmark_symbol,
        e.published_at.astimezone(UTC), e.received_at.astimezone(UTC),
        e.available_at.astimezone(UTC),
    )
    result = study_intraday([utc_event], minute_series("XNYS"))[0]
    assert result.status == "COMPLETED"
    assert result.entry_at == "2026-11-16T15:16:00+00:00"


def test_intraday_csv_rejects_raw_unadjusted_prices(tmp_path: Path) -> None:
    news = tmp_path / "news.csv"
    prices = tmp_path / "prices.csv"
    news.write_text(
        "event_id,market,symbol,benchmark_symbol,published_at,received_at,available_at\n"
        "7,XNAS,NVDA,SPY,2026-11-16T10:15:00-05:00,"
        "2026-11-16T10:15:30-05:00,2026-11-16T10:15:45-05:00\n",
        encoding="utf-8",
    )
    prices.write_text(
        "market,symbol,close_at,adjusted_close,adjustment\n"
        "XNAS,NVDA,2026-11-16T10:16:00-05:00,100,unadjusted\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unadjusted"):
        intraday_csv(news, prices)


def test_intraday_csv_with_two_sides_and_no_source_mutation(tmp_path: Path) -> None:
    news = tmp_path / "news.csv"
    prices = tmp_path / "prices.csv"
    news.write_text(
        "event_id,market,symbol,benchmark_symbol,published_at,received_at,available_at\n"
        "7,XNAS,NVDA,SPY,2026-11-16T10:15:00-05:00,"
        "2026-11-16T10:15:30-05:00,2026-11-16T10:15:45-05:00\n",
        encoding="utf-8",
    )
    prices.write_text(
        "market,symbol,close_at,adjusted_close,adjustment\n"
        "XNAS,NVDA,2026-11-16T10:16:00-05:00,100,adjusted\n"
        "XNAS,NVDA,2026-11-16T10:17:00-05:00,101,adjusted\n"
        "XNAS,SPY,2026-11-16T10:16:00-05:00,100,adjusted\n"
        "XNAS,SPY,2026-11-16T10:17:00-05:00,100.5,adjusted\n",
        encoding="utf-8",
    )
    originals = news.read_bytes(), prices.read_bytes()
    report = intraday_csv(news, prices)
    assert report["completed_label_count"] == 1
    assert report["prediction_model_trained"] is False
    assert report["causal_effect_estimated"] is False
    assert report["items"][0]["labels"][0]["excess_return"] == pytest.approx(0.005)
    assert originals == (news.read_bytes(), prices.read_bytes())
