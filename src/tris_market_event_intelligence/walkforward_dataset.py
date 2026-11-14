"""Research-only, point-in-time intraday samples with purged temporal evaluation.

Never treat ex-post labels as features. Samples inherit the earlier news-link
and session screens, and must not be deployed for trading or model inference.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from math import isfinite
from zoneinfo import ZoneInfo

from tris_market_event_intelligence.event_ticker_screen import (
    EventTickerLink,
    source_identity,
)
from tris_market_event_intelligence.exchange_sessions import MARKET_ZONES, utc
from tris_market_event_intelligence.intraday_event_study import (
    IntradayNews,
    IntradayResult,
)


@dataclass(frozen=True)
class ResearchSample:
    event_id: int
    market: str
    symbol: str
    horizon_minutes: int
    available_at: str
    label_exit_at: str
    split: str
    # Only values available by the event's inference time are input features.
    features: dict[str, str | int | float]
    # This is an ex-post historical training target, NEVER an input feature.
    target_excess_return: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def build_walkforward_dataset(
    events: Iterable[IntradayNews],
    links: Iterable[EventTickerLink],
    screened: Iterable[IntradayResult],
    *,
    cutoff: datetime,
    embargo_minutes: int = 60,
    horizons: tuple[int, ...] = (1, 5, 15, 30, 60),
) -> dict[str, object]:
    """Create purged train/holdout samples, excluding unsupported results.

    Embargo must cover the longest possible horizon. A training label must
    have fully matured BEFORE the cutoff minus embargo. The test event must
    become observable AT or AFTER cutoff plus embargo. No random splits.
    """
    split_at = utc(cutoff)
    if not horizons or any(type(h) is not int or h < 1 for h in horizons):
        raise ValueError("Expected positive, finite minute horizons")
    if len(set(horizons)) != len(horizons):
        raise ValueError("Repeated horizons are not permitted")
    if type(embargo_minutes) is not int or embargo_minutes < max(horizons):
        raise ValueError("Embargo must cover the maximum label horizon")
    event_rows, link_rows, results = list(events), list(links), list(screened)
    if len(event_rows) != len(results):
        raise ValueError("News/results row counts differ")
    keys: set[tuple[int, str, str]] = set()
    by_link: dict[tuple[int, str, str], EventTickerLink] = {}
    for link in link_rows:
        key = link.event_id, link.market, link.symbol
        if key in by_link:
            raise ValueError("Duplicate ticker link")
        by_link[key] = link
    for event, result in zip(event_rows, results, strict=True):
        key = event.event_id, event.market, event.symbol
        if key in keys or key != (
            result.event_id, result.market, result.symbol
        ):
            raise ValueError("Non-unique or mismatched news/results")
        keys.add(key)
    if keys != set(by_link):
        raise ValueError("Every news event requires exactly one matching link")
    # Never silently split duplicate stories across train and holdout.
    # Pick the earliest observable version regardless of input ordering.
    earliest: dict[tuple[str, str, str], tuple[datetime, int]] = {}
    for event in event_rows:
        link = by_link[(event.event_id, event.market, event.symbol)]
        article = source_identity(link.source_url)
        if article is None:
            raise ValueError("Invalid source identity")
        group = event.market, event.symbol, article
        candidate = utc(event.available_at), event.event_id
        if group not in earliest or candidate < earliest[group]:
            earliest[group] = candidate
    included: list[ResearchSample] = []
    counts: Counter[str] = Counter()
    left_limit = split_at - timedelta(minutes=embargo_minutes)
    right_limit = split_at + timedelta(minutes=embargo_minutes)
    for event, result in zip(event_rows, results, strict=True):
        key = event.event_id, event.market, event.symbol
        link = by_link[key]
        available = utc(event.available_at)
        if utc(link.evidence_seen_at) > available:
            raise ValueError("Future-discovered ticker relationship cannot become a feature")
        if result.benchmark_symbol != event.benchmark_symbol or (
            link.benchmark_symbol != event.benchmark_symbol
        ):
            raise ValueError("Benchmark mapping mismatch")
        article = source_identity(link.source_url)
        if article is None:
            raise ValueError("Invalid source identity")
        grouping = (event.market, event.symbol, article)
        if (available, event.event_id) != earliest[grouping]:
            counts["DUPLICATE_SOURCE"] += len(result.labels)
            continue
        if result.status not in {"COMPLETED", "PARTIAL"}:
            counts[result.status] += len(result.labels)
            continue
        if len(result.labels) != len(horizons) or {
            label.horizon_minutes for label in result.labels
        } != set(horizons):
            raise ValueError("Result labels do not match the declared horizon contract")
        if result.entry_at is None:
            raise ValueError("Eligible result must provide entry timestamp")
        entry_time = utc(datetime.fromisoformat(result.entry_at))
        expected_entry = available.replace(second=0, microsecond=0) + timedelta(minutes=1)
        if entry_time != expected_entry:
            raise ValueError("Entry timestamp must equal first post-availability minute")
        # Temporal provenance is checked for every sample, including
        # times crossing US daylight saving or Korea's midnight UTC boundary.
        for label in result.labels:
            if label.status != "COMPLETED" or label.excess_return is None:
                counts[label.status] += 1
                continue
            if label.exit_at is None:
                raise ValueError("Completed label must have an exit timestamp")
            exit_at = utc(datetime.fromisoformat(label.exit_at))
            if exit_at != entry_time + timedelta(minutes=label.horizon_minutes):
                raise ValueError("Invalid historical target timestamp relative to availability")
            if not isfinite(label.excess_return):
                raise ValueError("Nonfinite target is not a valid historical return")
            if available < split_at and exit_at < left_limit:
                split = "TRAIN"
            elif available >= right_limit:
                split = "HOLDOUT"
            else:
                counts["PURGED_TIME_BOUNDARY"] += 1
                continue
            # Features are explicitly allowlisted. No label, exit price, future
            # price, event ID or post-event model rationale goes into the features.
            local_time = available.astimezone(ZoneInfo(MARKET_ZONES[event.market]))
            features: dict[str, str | int | float] = {
                "market": event.market,
                "symbol": event.symbol,
                "relationship": link.relationship,
                "hour_of_day": local_time.hour,
                "weekday": local_time.weekday(),
                "source_latency_seconds": (
                    available - utc(event.published_at)
                ).total_seconds(),
                "horizon_minutes": label.horizon_minutes,
            }
            included.append(ResearchSample(
                event.event_id, event.market, event.symbol,
                label.horizon_minutes, available.isoformat(), exit_at.isoformat(),
                split, features, label.excess_return,
            ))
            counts[split] += 1
    return {
        "schema_version": "walkforward_dataset_v1",
        "cutoff_utc": split_at.isoformat(),
        "embargo_minutes": embargo_minutes,
        "horizons_minutes": list(horizons),
        "counts": dict(sorted(counts.items())),
        "samples": [row.as_dict() for row in included],
        "prediction_model_trained": False,
        "training_dataset_eligible": False,
        "point_in_time_verified": False,
        "note": (
            "Research-only syntactic data split. Source linkage, corporate-action "
            "adjustments, exchange calendar/halts and market data provenance are "
            "not independently verified. No predictive accuracy claim."
        ),
    }
