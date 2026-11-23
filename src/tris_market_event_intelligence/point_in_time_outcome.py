"""Point-in-time market outcome contract for news event research.

A market return is an observed price movement, never a causal news label.
This module deliberately does not infer a trading calendar, source availability,
or a first-publication timestamp from an article title or detected timestamp.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from math import isfinite
from typing import Literal

Novelty = Literal["NEW", "REPRINT", "FOLLOW_UP", "CORRECTION", "UNKNOWN"]
ForecastMode = Literal["LIVE", "HISTORICAL_REPLAY"]
ObservationStatus = Literal[
    "OBSERVED", "UNVERIFIED_SOURCE", "MISSING_BASE_PRICE",
    "MISSING_EXIT_PRICE", "MISSING_FORECAST_BASE", "MISSING_BENCHMARK", "STALE_PRICE",
]
Direction = Literal["UP", "DOWN", "NEUTRAL"]


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("All times must include a timezone offset")
    from datetime import UTC
    return value.astimezone(UTC)


@dataclass(frozen=True)
class EventClock:
    article_id: int
    family_id: str
    published_at: datetime
    first_public_at: datetime | None
    detected_at: datetime
    novelty: Novelty
    source_url: str
    source_verified: bool

    def __post_init__(self) -> None:
        if self.article_id <= 0 or not self.family_id.strip() or not self.source_url.strip():
            raise ValueError("Event identity and source are required")
        if self.novelty not in {"NEW", "REPRINT", "FOLLOW_UP", "CORRECTION", "UNKNOWN"}:
            raise ValueError("Unsupported article relationship")
        pub = _utc(self.published_at)
        detected = _utc(self.detected_at)
        if self.first_public_at is not None:
            first = _utc(self.first_public_at)
            if first > detected:
                raise ValueError("Verified first-public time cannot be later than detection")
        if pub > detected:
            raise ValueError("Publisher timestamp is later than first detection")

    @property
    def verified_cutoff(self) -> datetime | None:
        if not self.source_verified or self.first_public_at is None:
            return None
        return _utc(self.first_public_at)


@dataclass(frozen=True)
class TimestampedPrice:
    symbol: str
    bar_end: datetime
    adjusted_price: float
    provider: str
    source_id: str
    session_id: str

    def __post_init__(self) -> None:
        _utc(self.bar_end)
        if not all((self.symbol.strip(), self.provider.strip(),
                    self.source_id.strip(), self.session_id.strip())):
            raise ValueError("Price bars require symbol, source and session provenance")
        if not isfinite(self.adjusted_price) or self.adjusted_price <= 0:
            raise ValueError("Price must be positive finite")


@dataclass(frozen=True)
class SealedForecast:
    event: EventClock
    target: str
    benchmark: str
    direction: Direction
    forecast_at: datetime
    horizon_end: datetime
    horizon_name: str
    mode: ForecastMode
    model_version: str
    rule_version: str
    prediction_id: str
    deadband: float = 0.001

    def __post_init__(self) -> None:
        cutoff = _utc(self.forecast_at)
        end = _utc(self.horizon_end)
        if end <= cutoff:
            raise ValueError("Horizon must end strictly after sealed forecast")
        if self.mode not in {"LIVE", "HISTORICAL_REPLAY"}:
            raise ValueError("Unsupported forecast mode")
        if self.mode == "LIVE" and cutoff < _utc(self.event.detected_at):
            raise ValueError("Live forecast cannot precede article detection")
        if self.event.verified_cutoff is not None and cutoff < self.event.verified_cutoff:
            raise ValueError("Forecast cannot precede verified public availability")
        if self.target == self.benchmark or not all((
            self.target.strip(), self.benchmark.strip(), self.horizon_name.strip(),
            self.model_version.strip(), self.rule_version.strip(), self.prediction_id.strip(),
        )):
            raise ValueError("Forecast identity, provenance and distinct benchmark required")
        if self.direction not in {"UP", "DOWN", "NEUTRAL"}:
            raise ValueError("Unsupported market forecast")
        if not isfinite(self.deadband) or self.deadband < 0 or self.deadband >= 1:
            raise ValueError("Invalid deadband")


@dataclass(frozen=True)
class PriceObservation:
    prediction_id: str
    article_id: int
    family_id: str
    novelty: Novelty
    status: ObservationStatus
    anchor_at: str | None
    exit_at: str | None
    absolute_return: float | None
    benchmark_return: float | None
    relative_return: float | None
    actual_direction: Direction | None
    direction_match: bool | None
    forecast_anchor_at: str | None
    forecast_return: float | None
    forecast_actual_direction: Direction | None
    event_window_direction_match: bool | None
    causal_attribution: str = "NOT_ESTABLISHED"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _direction(value: float, deadband: float) -> Direction:
    if value > deadband:
        return "UP"
    if value < -deadband:
        return "DOWN"
    return "NEUTRAL"


def _nearest_before(
    prices: list[TimestampedPrice], symbol: str, boundary: datetime,
) -> TimestampedPrice | None:
    candidates = [p for p in prices if p.symbol == symbol and _utc(p.bar_end) < boundary]
    return max(candidates, key=lambda x: _utc(x.bar_end), default=None)


def _nearest_at_or_before(
    prices: list[TimestampedPrice], symbol: str, boundary: datetime,
    strictly_after: datetime,
) -> TimestampedPrice | None:
    candidates = [
        p for p in prices if p.symbol == symbol
        and strictly_after < _utc(p.bar_end) <= boundary
    ]
    return max(candidates, key=lambda x: _utc(x.bar_end), default=None)


def observe(
    forecast: SealedForecast,
    prices: list[TimestampedPrice],
    *,
    max_anchor_age: timedelta,
    max_exit_age: timedelta,
    benchmark_tolerance: timedelta = timedelta(minutes=5),
) -> PriceObservation:
    """Evaluate an already sealed forecast using real timestamped price bars.

    The anchor is the latest completed bar STRICTLY BEFORE first public availability,
    never the day's open or a same-day close. The exit is the last completed bar at
    or before the predeclared horizon. A stale/missing bar is NOT a zero return.
    Returns do not identify news causality and repeated articles remain tagged.
    """
    if max_anchor_age <= timedelta(0) or max_exit_age <= timedelta(0):
        raise ValueError("Positive maximum observation ages required")
    if benchmark_tolerance < timedelta(0):
        raise ValueError("Benchmark tolerance cannot be negative")
    event = forecast.event
    base = dict(prediction_id=forecast.prediction_id, article_id=event.article_id,
                family_id=event.family_id, novelty=event.novelty)
    def unavailable(status: ObservationStatus) -> PriceObservation:
        return PriceObservation(**base, status=status, anchor_at=None, exit_at=None,
                                absolute_return=None, benchmark_return=None,
                                relative_return=None, actual_direction=None,
                                direction_match=None, forecast_anchor_at=None,
                                forecast_return=None, forecast_actual_direction=None,
                                event_window_direction_match=None)

    anchor_time = event.verified_cutoff
    if anchor_time is None:
        return unavailable("UNVERIFIED_SOURCE")
    cutoff = _utc(forecast.horizon_end)
    prediction_at = _utc(forecast.forecast_at)
    target_base = _nearest_before(prices, forecast.target, anchor_time)
    if target_base is None:
        return unavailable("MISSING_BASE_PRICE")
    target_exit = _nearest_at_or_before(prices, forecast.target, cutoff, prediction_at)
    if target_exit is None:
        return unavailable("MISSING_EXIT_PRICE")
    forecast_base = _nearest_at_or_before(
        prices, forecast.target, prediction_at, anchor_time - timedelta(microseconds=1)
    )
    if forecast_base is None:
        return unavailable("MISSING_FORECAST_BASE")
    if prediction_at - _utc(forecast_base.bar_end) > max_anchor_age:
        return unavailable("STALE_PRICE")
    if (anchor_time - _utc(target_base.bar_end) > max_anchor_age
            or cutoff - _utc(target_exit.bar_end) > max_exit_age):
        return unavailable("STALE_PRICE")

    benchmark_base = _nearest_before(prices, forecast.benchmark, anchor_time)
    benchmark_exit = _nearest_at_or_before(prices, forecast.benchmark, cutoff, prediction_at)
    benchmark_forecast_base = _nearest_at_or_before(
        prices, forecast.benchmark, prediction_at, anchor_time - timedelta(microseconds=1)
    )
    if benchmark_base is None or benchmark_exit is None or benchmark_forecast_base is None:
        return unavailable("MISSING_BENCHMARK")
    if (abs(_utc(benchmark_base.bar_end) - _utc(target_base.bar_end))
            > benchmark_tolerance
            or abs(_utc(benchmark_exit.bar_end) - _utc(target_exit.bar_end))
            > benchmark_tolerance):
        return unavailable("MISSING_BENCHMARK")
    if abs(_utc(benchmark_forecast_base.bar_end) - _utc(forecast_base.bar_end)) > benchmark_tolerance:
        return unavailable("MISSING_BENCHMARK")
    if prediction_at - _utc(benchmark_forecast_base.bar_end) > max_anchor_age:
        return unavailable("STALE_PRICE")
    if (anchor_time - _utc(benchmark_base.bar_end) > max_anchor_age
            or cutoff - _utc(benchmark_exit.bar_end) > max_exit_age):
        return unavailable("STALE_PRICE")

    target_return = target_exit.adjusted_price / target_base.adjusted_price - 1.0
    benchmark_return = benchmark_exit.adjusted_price / benchmark_base.adjusted_price - 1.0
    actual = _direction(target_return, forecast.deadband)
    forecast_return = target_exit.adjusted_price / forecast_base.adjusted_price - 1.0
    forecast_actual = _direction(forecast_return, forecast.deadband)
    return PriceObservation(
        **base, status="OBSERVED", anchor_at=_utc(target_base.bar_end).isoformat(),
        exit_at=_utc(target_exit.bar_end).isoformat(),
        absolute_return=target_return, benchmark_return=benchmark_return,
        relative_return=target_return - benchmark_return, actual_direction=actual,
        direction_match=forecast_actual == forecast.direction,
        forecast_anchor_at=_utc(forecast_base.bar_end).isoformat(),
        forecast_return=forecast_return,
        forecast_actual_direction=forecast_actual,
        event_window_direction_match=actual == forecast.direction,
    )
