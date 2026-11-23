"""Point-in-time outcome regression: real price arithmetic on synthetic test fixtures.

These are controlled fixtures, not claims about historical real market accuracy.
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from tris_market_event_intelligence.point_in_time_outcome import (
    EventClock,
    SealedForecast,
    TimestampedPrice,
    observe,
)

KST = ZoneInfo("Asia/Seoul")


def dt(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 11, day, hour, minute, tzinfo=KST)


def event(*, novelty: str = "NEW", first: datetime | None = None,
          verified: bool = True, article: int = 1) -> EventClock:
    return EventClock(
        article_id=article, family_id="issuer-earnings-20261123",
        published_at=dt(23, 14), first_public_at=first if first is not None else dt(23, 14),
        detected_at=dt(23, 14, 1), novelty=novelty,
        source_url="https://example.org/issuer/1", source_verified=verified,
    )


def forecast(e: EventClock | None = None, *, direction: str = "DOWN",
             end: datetime | None = None, mode: str = "LIVE") -> SealedForecast:
    return SealedForecast(
        event=e or event(), target="EQUITY", benchmark="INDEX",
        direction=direction, forecast_at=dt(23, 14, 2),
        horizon_end=end or dt(23, 15, 30), horizon_name="session-close",
        mode=mode, model_version="gemma4:e2b", rule_version="none",
        prediction_id="sealed:1",
    )


def bar(symbol: str, time: datetime, price: float) -> TimestampedPrice:
    return TimestampedPrice(symbol=symbol, bar_end=time, adjusted_price=price,
                            provider="sample", source_id="fixture-001", session_id="20261123")


def default_bars() -> list[TimestampedPrice]:
    return [
        bar("EQUITY", dt(23, 9), 100),
        bar("EQUITY", dt(23, 13, 59), 110),
        bar("EQUITY", dt(23, 14), 111),
        bar("EQUITY", dt(23, 15, 30), 102),
        bar("INDEX", dt(23, 13, 59), 100),
        bar("INDEX", dt(23, 14), 100),
        bar("INDEX", dt(23, 15, 30), 99),
    ]


def evaluate(f: SealedForecast, bars: list[TimestampedPrice] | None = None):
    return observe(
        f, default_bars() if bars is None else bars,
        max_anchor_age=timedelta(minutes=5),
        max_exit_age=timedelta(minutes=5),
    )


def test_1400_bad_news_is_down_even_if_day_closes_plus_two_percent() -> None:
    result = evaluate(forecast())
    assert result.status == "OBSERVED"
    assert result.actual_direction == "DOWN"
    assert result.direction_match is True
    assert result.absolute_return == pytest.approx(102 / 110 - 1)
    assert result.relative_return == pytest.approx(102 / 110 - 1 - (99 / 100 - 1))
    assert result.anchor_at == dt(23, 13, 59).astimezone(
        ZoneInfo("UTC")).isoformat()
    assert result.causal_attribution == "NOT_ESTABLISHED"


def test_reprint_rebound_keeps_novelty_and_does_not_claim_causality() -> None:
    repeat = event(novelty="REPRINT", article=2)
    result = evaluate(forecast(repeat, direction="UP"), [
        bar("EQUITY", dt(23, 13, 59), 92),
        bar("EQUITY", dt(23, 15, 30), 95),
        bar("INDEX", dt(23, 13, 59), 100),
        bar("INDEX", dt(23, 15, 30), 101),
    ])
    assert result.novelty == "REPRINT"
    assert result.family_id == "issuer-earnings-20261123"
    assert result.direction_match is True
    assert result.causal_attribution == "NOT_ESTABLISHED"


def test_after_close_requires_explicit_session_anchor_and_future_window() -> None:
    after = EventClock(
        article_id=3, family_id="f3", published_at=dt(23, 16),
        first_public_at=dt(23, 16), detected_at=dt(23, 16, 1),
        novelty="NEW", source_url="https://example.org/3", source_verified=True,
    )
    f = SealedForecast(after, "EQUITY", "INDEX", "DOWN", dt(23, 16, 2),
                       dt(24, 15, 30), "next-session", "LIVE", "m1", "v1", "s3")
    bars = [
        bar("EQUITY", dt(23, 15, 30), 100),
        bar("INDEX", dt(23, 15, 30), 100),
        bar("EQUITY", dt(24, 15, 30), 98),
        bar("INDEX", dt(24, 15, 30), 101),
    ]
    result = observe(f, bars, max_anchor_age=timedelta(hours=3),
                     max_exit_age=timedelta(minutes=5))
    assert result.status == "OBSERVED"
    assert result.direction_match is True


def test_stale_before_news_price_does_not_become_zero_or_success() -> None:
    bars = [
        bar("EQUITY", dt(23, 9), 100),
        bar("EQUITY", dt(23, 15, 30), 102),
        bar("INDEX", dt(23, 9), 100),
        bar("INDEX", dt(23, 15, 30), 99),
    ]
    result = evaluate(forecast(), bars)
    assert result.status == "STALE_PRICE"
    assert result.direction_match is None
    assert result.absolute_return is None


def test_missing_market_benchmark_is_not_falsely_scored() -> None:
    result = evaluate(forecast(), [
        bar("EQUITY", dt(23, 13, 59), 110),
        bar("EQUITY", dt(23, 15, 30), 102),
    ])
    assert result.status == "MISSING_BENCHMARK"
    assert result.actual_direction is None


def test_unverified_original_publication_cannot_produce_score() -> None:
    result = evaluate(forecast(event(verified=False)))
    assert result.status == "UNVERIFIED_SOURCE"
    assert result.absolute_return is None


def test_later_forecast_cannot_use_price_that_predates_forecast_as_outcome() -> None:
    f = forecast(end=dt(23, 14, 5))
    result = evaluate(f, [
        bar("EQUITY", dt(23, 13, 59), 110),
        bar("EQUITY", dt(23, 14), 107),
        bar("INDEX", dt(23, 13, 59), 100),
        bar("INDEX", dt(23, 14), 100),
    ])
    assert result.status == "MISSING_EXIT_PRICE"


def test_bad_timestamps_and_unsealed_live_forecast_are_rejected() -> None:
    with pytest.raises(ValueError):
        EventClock(1, "family", dt(23, 14), dt(23, 14, 5), dt(23, 14, 1),
                   "NEW", "source", True)
    with pytest.raises(ValueError):
        forecast(event(), end=dt(23, 14, 1))
    with pytest.raises(ValueError):
        TimestampedPrice("T", datetime(2026, 11, 23, 14), 1, "a", "b", "c")


def test_price_bar_exactly_at_publication_not_used_as_before_news_anchor() -> None:
    result = evaluate(forecast(), [
        bar("EQUITY", dt(23, 14), 110),
        bar("EQUITY", dt(23, 15, 30), 102),
        bar("INDEX", dt(23, 14), 100),
        bar("INDEX", dt(23, 15, 30), 99),
    ])
    assert result.status == "MISSING_BASE_PRICE"


def test_forecast_is_not_economic_meaning_accuracy() -> None:
    result = evaluate(forecast(direction="UP"))
    assert result.status == "OBSERVED"
    assert result.actual_direction == "DOWN"
    assert result.direction_match is False
    assert "economic" not in result.as_dict()
