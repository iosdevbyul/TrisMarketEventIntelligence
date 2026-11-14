"""Cross-market regular-session intraday event study, CPU-only and fail-closed.

Minute bars are close-time stamped *adjusted* prices. Only minutes after the
information was actually available are used. Exact clock-minute horizons,
never N observed rows, avoid compressing missing intervals or session gaps.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, time, timedelta
from math import isfinite
from zoneinfo import ZoneInfo

from tris_market_event_intelligence.exchange_sessions import SessionCalendar

INTRADAY_HORIZONS = (1, 5, 15, 30, 60)
MARKETS = {
    "XKRX": ("Asia/Seoul", time(9, 0), time(15, 30)),
    "XNYS": ("America/New_York", time(9, 30), time(16, 0)),
    "XNAS": ("America/New_York", time(9, 30), time(16, 0)),
}
_ONE_MINUTE = timedelta(minutes=1)


@dataclass(frozen=True)
class IntradayNews:
    event_id: int
    market: str
    symbol: str
    benchmark_symbol: str
    published_at: datetime
    received_at: datetime
    available_at: datetime


@dataclass(frozen=True)
class MinuteClose:
    market: str
    symbol: str
    close_at: datetime
    adjusted_close: float


@dataclass(frozen=True)
class IntradayLabel:
    horizon_minutes: int
    exit_at: str | None
    stock_return: float | None
    benchmark_return: float | None
    excess_return: float | None
    status: str


@dataclass(frozen=True)
class IntradayResult:
    event_id: int
    market: str
    symbol: str
    benchmark_symbol: str
    entry_at: str | None
    status: str
    latency_seconds: float
    labels: tuple[IntradayLabel, ...]

    def as_dict(self) -> dict[str, object]:
        return {**asdict(self), "labels": [asdict(label) for label in self.labels]}


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("All timestamps must have an explicit timezone")
    return value.astimezone(UTC)


def _regular_session(
    market: str, instant: datetime, *, include_open: bool = False,
) -> bool:
    zone, start, end = MARKETS[market]
    local = instant.astimezone(ZoneInfo(zone))
    clock = local.timetz().replace(tzinfo=None)
    return local.weekday() < 5 and (
        start <= clock <= end if include_open else start < clock <= end
    )


def _prepare(
    bars: Iterable[MinuteClose], calendar: SessionCalendar | None = None,
) -> dict[tuple[str, str], dict[datetime, float]]:
    result: dict[tuple[str, str], dict[datetime, float]] = {}
    for bar in bars:
        if bar.market not in MARKETS or not bar.symbol.strip():
            raise ValueError("Minute close requires a supported market and symbol")
        stamp = _utc(bar.close_at)
        if stamp.second != 0 or stamp.microsecond != 0:
            raise ValueError("Minute bars require exact minute-end timestamps")
        status = (
            calendar.instant_status(bar.market, stamp)
            if calendar is not None else
            ("IN_SESSION" if _regular_session(bar.market, stamp) else "OUTSIDE_REGULAR_SESSION")
        )
        if status != "IN_SESSION":
            raise ValueError(f"Minute bar outside confirmed exchange session: {status}")
        if not isfinite(bar.adjusted_close) or bar.adjusted_close <= 0:
            raise ValueError("Minute closes must have positive finite adjusted prices")
        series = result.setdefault((bar.market, bar.symbol), {})
        if stamp in series:
            raise ValueError("Duplicate minute close timestamp")
        series[stamp] = bar.adjusted_close
    return result


def study_intraday(
    events: Iterable[IntradayNews],
    minute_closes: Iterable[MinuteClose],
    *, horizons: tuple[int, ...] = INTRADAY_HORIZONS,
    calendar: SessionCalendar | None = None,
) -> list[IntradayResult]:
    """Produce descriptive minute-price labels without future-information leakage."""
    if not horizons or any(type(h) is not int or h < 1 for h in horizons):
        raise ValueError("Horizons must be positive integer minutes")
    if len(set(horizons)) != len(horizons):
        raise ValueError("Horizons must be unique")
    prices = _prepare(minute_closes, calendar)
    observed: set[tuple[int, str, str]] = set()
    results: list[IntradayResult] = []
    for event in events:
        if event.event_id < 1 or event.market not in MARKETS:
            raise ValueError("Event requires positive ID and supported market")
        if not event.symbol.strip() or not event.benchmark_symbol.strip():
            raise ValueError("Event requires symbols for stock and benchmark")
        if event.symbol == event.benchmark_symbol:
            raise ValueError("Benchmark must be distinct from the stock symbol")
        key = event.event_id, event.market, event.symbol
        if key in observed:
            raise ValueError("Duplicate event-market-symbol")
        observed.add(key)
        published, received, available = (
            _utc(event.published_at), _utc(event.received_at), _utc(event.available_at)
        )
        if not published <= received <= available:
            raise ValueError("Publication, receipt and availability must be chronological")
        latency = (available - published).total_seconds()

        def missing(
            reason: str, entry: datetime | None = None,
            source: IntradayNews = event, delay: float = latency,
        ) -> IntradayResult:
            return IntradayResult(
                source.event_id, source.market, source.symbol, source.benchmark_symbol,
                entry.isoformat() if entry else None, reason, delay,
                tuple(IntradayLabel(h, None, None, None, None, reason) for h in horizons),
            )

        available_status = (
            calendar.instant_status(event.market, available, include_open=True)
            if calendar is not None else
            ("IN_SESSION" if _regular_session(event.market, available, include_open=True)
             else "OUTSIDE_REGULAR_SESSION")
        )
        if available_status != "IN_SESSION":
            results.append(missing(available_status))
            continue

        # The price at exactly available_at might not yet have been received;
        # choose the first CLOSED minute strictly later than information availability.
        entry = available.replace(second=0, microsecond=0) + _ONE_MINUTE
        entry_status = (
            calendar.instant_status(event.market, entry)
            if calendar is not None else
            ("IN_SESSION" if _regular_session(event.market, entry)
             else "NO_ENTRY_IN_SESSION")
        )
        if entry_status != "IN_SESSION":
            results.append(missing(
                "NO_ENTRY_IN_SESSION" if entry_status == "OUTSIDE_REGULAR_SESSION"
                else entry_status,
            ))
            continue
        if calendar is not None and calendar.intersects_halt(
            event.market, (event.symbol, event.benchmark_symbol), entry, entry,
        ):
            results.append(missing("TRADING_HALTED_AT_ENTRY", entry))
            continue
        stock = prices.get((event.market, event.symbol), {})
        bench = prices.get((event.market, event.benchmark_symbol), {})
        if entry not in stock:
            results.append(missing("STOCK_ENTRY_MISSING", entry))
            continue
        if entry not in bench:
            results.append(missing("BENCHMARK_ENTRY_MISSING", entry))
            continue

        labels: list[IntradayLabel] = []
        for horizon in horizons:
            exit_at = entry + timedelta(minutes=horizon)
            exit_iso = exit_at.isoformat()
            exit_status = (
                calendar.instant_status(event.market, exit_at)
                if calendar is not None else
                ("IN_SESSION" if _regular_session(event.market, exit_at)
                 else "OUTSIDE_REGULAR_SESSION")
            )
            if exit_status != "IN_SESSION":
                labels.append(IntradayLabel(
                    horizon, exit_iso, None, None, None,
                    "SESSION_BOUNDARY" if exit_status == "OUTSIDE_REGULAR_SESSION"
                    else exit_status,
                ))
                continue
            if calendar is not None and calendar.intersects_halt(
                event.market, (event.symbol, event.benchmark_symbol), entry, exit_at,
            ):
                labels.append(IntradayLabel(
                    horizon, exit_iso, None, None, None, "TRADING_HALTED_WINDOW",
                ))
                continue
            # No missing bar compression: every minute must have a recorded
            # close for both stock and benchmark, or the label is unavailable.
            timeline = (entry + timedelta(minutes=offset) for offset in range(1, horizon + 1))
            if any(moment not in stock or moment not in bench for moment in timeline):
                labels.append(IntradayLabel(
                    horizon, exit_iso, None, None, None, "MISSING_MINUTE_BAR",
                ))
                continue
            stock_return = stock[exit_at] / stock[entry] - 1.0
            market_return = bench[exit_at] / bench[entry] - 1.0
            labels.append(IntradayLabel(
                horizon, exit_iso, stock_return, market_return,
                stock_return - market_return, "COMPLETED",
            ))
        results.append(IntradayResult(
            event.event_id, event.market, event.symbol, event.benchmark_symbol,
            entry.isoformat(), "COMPLETED" if all(
                label.status == "COMPLETED" for label in labels
            ) else "PARTIAL", latency, tuple(labels),
        ))
    return results
