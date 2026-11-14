"""Leakage-resistant, CPU-only historical event-study label computation.

The first bar strictly AFTER publication is the entry close. Horizons are
subsequent trading-session closes; this does not estimate contemporaneous
price impact or trade execution returns. Market-adjusted returns are
descriptive historical labels, not identified causal effects.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from math import isfinite
from typing import Iterable

HORIZONS = (1, 5, 20)


@dataclass(frozen=True)
class DailyClose:
    symbol: str
    day: date
    adjusted_close: float


@dataclass(frozen=True)
class PublishedEvent:
    event_id: int
    symbol: str
    published_at: datetime


@dataclass(frozen=True)
class HorizonLabel:
    horizon_sessions: int
    exit_day: str | None
    stock_return: float | None
    benchmark_return: float | None
    excess_return: float | None
    status: str


@dataclass(frozen=True)
class EventStudyResult:
    event_id: int
    symbol: str
    benchmark_symbol: str
    entry_day: str | None
    status: str
    labels: tuple[HorizonLabel, ...]

    def as_dict(self) -> dict[str, object]:
        return {**asdict(self), "labels": [asdict(label) for label in self.labels]}


def _index(closes: Iterable[DailyClose]) -> dict[str, list[DailyClose]]:
    indexed: dict[str, dict[date, DailyClose]] = {}
    for bar in closes:
        if not bar.symbol.strip() or not isfinite(bar.adjusted_close) or bar.adjusted_close <= 0:
            raise ValueError("Expected positive finite adjusted closes and nonempty symbols")
        series = indexed.setdefault(bar.symbol, {})
        if bar.day in series:
            raise ValueError(f"Duplicate daily close: {bar.symbol} {bar.day}")
        series[bar.day] = bar
    return {symbol: [by_day[d] for d in sorted(by_day)] for symbol, by_day in indexed.items()}


def study_events(
    events: Iterable[PublishedEvent],
    closes: Iterable[DailyClose],
    *, benchmark_symbol: str, horizons: tuple[int, ...] = HORIZONS,
) -> list[EventStudyResult]:
    """Assign future, benchmark-aligned labels without inserting future bars."""
    if not benchmark_symbol.strip():
        raise ValueError("Benchmark symbol is required")
    if not horizons or any(type(h) is not int or h < 1 for h in horizons):
        raise ValueError("Horizons must be positive integer trading sessions")
    if len(set(horizons)) != len(horizons):
        raise ValueError("Horizons must be unique")
    prices = _index(closes)
    market = {bar.day: bar for bar in prices.get(benchmark_symbol, [])}
    results: list[EventStudyResult] = []
    seen: set[tuple[int, str]] = set()
    for event in events:
        if event.event_id < 1 or not event.symbol.strip():
            raise ValueError("Event IDs must be positive and symbols nonempty")
        if event.published_at.tzinfo is None or event.published_at.utcoffset() is None:
            raise ValueError("Publication time must be timezone-aware")
        key = (event.event_id, event.symbol)
        if key in seen:
            raise ValueError("Duplicate event-symbol pair")
        seen.add(key)

        # Use the event's local civil date, NOT UTC, to avoid shifting KST
        # publications across session boundaries. Always enter on a strictly
        # later trading date, conservatively discarding same-day reactions.
        publication_day = event.published_at.date()
        stock = prices.get(event.symbol, [])
        future = [bar for bar in stock if bar.day > publication_day]
        if not future:
            results.append(EventStudyResult(
                event.event_id, event.symbol, benchmark_symbol, None,
                "NO_ENTRY_SESSION", tuple(HorizonLabel(
                    h, None, None, None, None, "NO_ENTRY_SESSION"
                ) for h in horizons),
            ))
            continue
        entry = future[0]
        if entry.day not in market:
            results.append(EventStudyResult(
                event.event_id, event.symbol, benchmark_symbol, entry.day.isoformat(),
                "BENCHMARK_ENTRY_MISSING", tuple(HorizonLabel(
                    h, None, None, None, None, "BENCHMARK_ENTRY_MISSING"
                ) for h in horizons),
            ))
            continue
        base_market = market[entry.day]
        labels: list[HorizonLabel] = []
        for horizon in horizons:
            if horizon >= len(future):
                labels.append(HorizonLabel(horizon, None, None, None, None, "INSUFFICIENT_FUTURE"))
                continue
            exit_bar = future[horizon]
            if exit_bar.day not in market:
                labels.append(HorizonLabel(
                    horizon, exit_bar.day.isoformat(), None, None, None,
                    "BENCHMARK_EXIT_MISSING",
                ))
                continue
            market_exit = market[exit_bar.day]
            sr = exit_bar.adjusted_close / entry.adjusted_close - 1.0
            mr = market_exit.adjusted_close / base_market.adjusted_close - 1.0
            labels.append(HorizonLabel(
                horizon, exit_bar.day.isoformat(), sr, mr, sr - mr, "COMPLETED",
            ))
        results.append(EventStudyResult(
            event.event_id, event.symbol, benchmark_symbol, entry.day.isoformat(),
            "COMPLETED" if all(v.status == "COMPLETED" for v in labels) else "PARTIAL",
            tuple(labels),
        ))
    return results
