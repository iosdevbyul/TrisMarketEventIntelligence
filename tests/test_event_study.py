"""Event study historical labels must never use the publication day's closing price."""
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from tris_market_event_intelligence.event_study import (
    DailyClose,
    PublishedEvent,
    study_events,
)
from tris_market_event_intelligence.event_study_cli import study_csv

KST = timezone(timedelta(hours=9))


def observations():
    days = [
        date(2026, 11, 12), date(2026, 11, 13), date(2026, 11, 16),
        date(2026, 11, 17), date(2026, 11, 18), date(2026, 11, 19),
        date(2026, 11, 20), date(2026, 11, 23),
    ]
    return [
        DailyClose("005930", d, float(v))
        for d, v in zip(days, [100, 120, 90, 99, 108, 100, 115, 118], strict=True)
    ] + [
        DailyClose("KOSPI", d, float(v))
        for d, v in zip(days, [100, 105, 100, 102, 104, 106, 105, 110], strict=True)
    ]


def test_publication_day_not_used_and_future_sessions_aligned() -> None:
    event = PublishedEvent(13, "005930", datetime(2026, 11, 13, 11, 0, tzinfo=KST))
    result = study_events([event], observations(), benchmark_symbol="KOSPI")
    item = result[0]
    assert item.entry_day == "2026-11-16"
    assert item.status == "PARTIAL"
    one = item.labels[0]
    assert one.exit_day == "2026-11-17"
    assert one.stock_return == pytest.approx(0.10)
    assert one.benchmark_return == pytest.approx(0.02)
    assert one.excess_return == pytest.approx(0.08)
    assert item.labels[1].exit_day == "2026-11-23"
    assert item.labels[2].status == "INSUFFICIENT_FUTURE"
    assert item.labels[2].excess_return is None


def test_weekend_publication_enters_next_session_close() -> None:
    event = PublishedEvent(2, "005930", datetime(2026, 11, 14, 18, tzinfo=KST))
    result = study_events(
        [event], observations(), benchmark_symbol="KOSPI", horizons=(1,),
    )
    assert result[0].entry_day == "2026-11-16"


def test_no_future_entry_has_no_fabricated_labels() -> None:
    event = PublishedEvent(3, "005930", datetime(2026, 11, 24, 10, tzinfo=KST))
    result = study_events(
        [event], observations(), benchmark_symbol="KOSPI", horizons=(1, 5),
    )[0]
    assert result.status == "NO_ENTRY_SESSION"
    assert all(label.excess_return is None for label in result.labels)


@pytest.mark.parametrize("broken", [
    DailyClose("005930", date(2026, 11, 16), 0),
    DailyClose("005930", date(2026, 11, 16), float("nan")),
    DailyClose("005930", date(2026, 11, 16), -3),
])
def test_invalid_closes_rejected(broken: DailyClose) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        study_events([], [broken], benchmark_symbol="KOSPI")


def test_naive_publication_and_duplicate_prices_are_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        study_events(
            [PublishedEvent(1, "005930", datetime(2026, 11, 13))],
            observations(), benchmark_symbol="KOSPI",
        )
    bars = observations()
    with pytest.raises(ValueError, match="Duplicate daily close"):
        study_events([], bars + [bars[0]], benchmark_symbol="KOSPI")


def test_duplicate_events_horizons_and_missing_benchmark_rejected() -> None:
    e = PublishedEvent(7, "005930", datetime(2026, 11, 13, tzinfo=KST))
    with pytest.raises(ValueError, match="Duplicate event-symbol"):
        study_events([e, e], observations(), benchmark_symbol="KOSPI")
    with pytest.raises(ValueError, match="Horizons must be unique"):
        study_events([e], observations(), benchmark_symbol="KOSPI", horizons=(1, 1))
    assert study_events([e], observations(), benchmark_symbol="NONE")[0].status == (
        "BENCHMARK_ENTRY_MISSING"
    )


def test_csv_adapter_offline_and_read_only(tmp_path: Path) -> None:
    event_csv = tmp_path / "events.csv"
    prices_csv = tmp_path / "prices.csv"
    event_csv.write_text(
        "event_id,symbol,published_at\n13,005930,2026-11-13T09:00:00+09:00\n",
        encoding="utf-8",
    )
    prices_csv.write_text(
        "symbol,date,adjusted_close\n"
        "005930,2026-11-13,100\n005930,2026-11-16,90\n"
        "005930,2026-11-17,99\nKOSPI,2026-11-16,100\n"
        "KOSPI,2026-11-17,102\n",
        encoding="utf-8",
    )
    originals = event_csv.read_bytes(), prices_csv.read_bytes()
    output = study_csv(event_csv, prices_csv, benchmark="KOSPI")
    assert output["label_count"] == 1
    assert output["items"][0]["labels"][0]["excess_return"] == pytest.approx(0.08)
    assert output["quality_verified"] is False
    assert output["causal_effect_estimated"] is False
    assert (event_csv.read_bytes(), prices_csv.read_bytes()) == originals


def test_csv_adapter_rejects_missing_or_invalid_inputs(tmp_path: Path) -> None:
    event_csv = tmp_path / "events.csv"
    prices_csv = tmp_path / "prices.csv"
    event_csv.write_text("event_id,symbol\n13,005930\n", encoding="utf-8")
    prices_csv.write_text("symbol,date,adjusted_close\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required headers"):
        study_csv(event_csv, prices_csv, benchmark="KOSPI")


def test_missing_benchmark_exit_does_not_shift_horizon() -> None:
    event = PublishedEvent(8, "005930", datetime(2026, 11, 13, tzinfo=KST))
    closes = [
        bar for bar in observations()
        if not (bar.symbol == "KOSPI" and bar.day == date(2026, 11, 17))
    ]
    item = study_events(
        [event], closes, benchmark_symbol="KOSPI", horizons=(1, 2),
    )[0]
    assert item.labels[0].status == "BENCHMARK_EXIT_MISSING"
    assert item.labels[0].exit_day == "2026-11-17"
    assert item.labels[0].excess_return is None
    assert item.labels[1].status == "COMPLETED"
    assert item.labels[1].exit_day == "2026-11-18"


def test_utc_publication_converted_to_exchange_local_date() -> None:
    # Nov 13 16:30 UTC is already Nov 14 in KST, thus Monday is first entry.
    event = PublishedEvent(10, "005930", datetime(2026, 11, 13, 16, 30, tzinfo=UTC))
    item = study_events(
        [event], observations(), benchmark_symbol="KOSPI", horizons=(1,),
    )[0]
    assert item.entry_day == "2026-11-16"
