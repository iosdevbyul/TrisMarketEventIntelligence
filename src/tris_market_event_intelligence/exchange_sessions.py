"""Explicit, source-supplied exchange calendar and trading interruption contract.

No invented exchange holidays, special closes, or trading halts. A calendar
provides known OPEN/CLOSED dates; dates not provided are UNKNOWN, not closed.
All instants are timezone-aware and represented in absolute UTC.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from itertools import pairwise
from zoneinfo import ZoneInfo

MARKET_ZONES = {
    "XKRX": "Asia/Seoul",
    "XNYS": "America/New_York",
    "XNAS": "America/New_York",
}


def utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Exchange schedule timestamps require explicit timezone")
    return value.astimezone(UTC)


@dataclass(frozen=True)
class SessionDay:
    market: str
    day: date
    status: str
    open_at: datetime | None = None
    close_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.market not in MARKET_ZONES:
            raise ValueError("Unsupported exchange")
        if self.status not in {"OPEN", "CLOSED"}:
            raise ValueError("Session status must be OPEN or CLOSED")
        if self.status == "CLOSED":
            if self.open_at is not None or self.close_at is not None:
                raise ValueError("Closed session must not include opening or closing times")
            return
        if self.open_at is None or self.close_at is None:
            raise ValueError("Open session requires exact opening and closing times")
        start, end = utc(self.open_at), utc(self.close_at)
        if start >= end:
            raise ValueError("Session opening must precede closing")
        zone = ZoneInfo(MARKET_ZONES[self.market])
        if start.astimezone(zone).date() != self.day or end.astimezone(zone).date() != self.day:
            raise ValueError("Session bounds must use the declared exchange-local date")


@dataclass(frozen=True)
class TradingHalt:
    market: str
    symbol: str
    start_at: datetime
    end_at: datetime

    def __post_init__(self) -> None:
        if self.market not in MARKET_ZONES or not self.symbol.strip():
            raise ValueError("Halt requires supported exchange and symbol")
        if utc(self.start_at) >= utc(self.end_at):
            raise ValueError("Halt start must precede end")


class SessionCalendar:
    """Immutable, externally reviewed market-date entries and halt intervals."""

    def __init__(
        self, sessions: list[SessionDay], halts: list[TradingHalt] | None = None,
    ) -> None:
        if not sessions:
            raise ValueError("Calendar must have at least one explicitly reviewed day")
        self._days: dict[tuple[str, date], SessionDay] = {}
        for session in sessions:
            key = session.market, session.day
            if key in self._days:
                raise ValueError("Duplicate market calendar day")
            self._days[key] = session
        self._halts: dict[tuple[str, str], list[tuple[datetime, datetime]]] = {}
        for halt in halts or []:
            self._halts.setdefault((halt.market, halt.symbol), []).append(
                (utc(halt.start_at), utc(halt.end_at))
            )
        for intervals in self._halts.values():
            intervals.sort()
            if any(left[1] > right[0] for left, right in pairwise(intervals)):
                raise ValueError("Overlapping halt intervals for one exchange symbol")

    def session(self, market: str, instant: datetime) -> SessionDay | None:
        if market not in MARKET_ZONES:
            raise ValueError("Unsupported exchange")
        local_day = utc(instant).astimezone(ZoneInfo(MARKET_ZONES[market])).date()
        return self._days.get((market, local_day))

    def instant_status(
        self, market: str, instant: datetime, *, include_open: bool = False,
    ) -> str:
        entry = self.session(market, instant)
        if entry is None:
            return "CALENDAR_UNAVAILABLE"
        if entry.status == "CLOSED":
            return "MARKET_CLOSED"
        now = utc(instant)
        start, end = utc(entry.open_at), utc(entry.close_at)
        if start <= now <= end if include_open else start < now <= end:
            return "IN_SESSION"
        return "OUTSIDE_REGULAR_SESSION"

    def intersects_halt(
        self, market: str, symbols: tuple[str, ...], start: datetime, end: datetime,
    ) -> bool:
        """Inclusive price observation window intersects [halt_start, halt_end)."""
        left, right = utc(start), utc(end)
        if left > right:
            raise ValueError("Invalid observation interval")
        for symbol in symbols:
            for begin, finish in self._halts.get((market, symbol), []):
                if left < finish and right >= begin:
                    return True
        return False
