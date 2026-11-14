"""Purged, disjoint rolling walk-forward research evaluation on Intel CPUs.

Rebuild each split from the *original* screened incidents, never from a
previously purged single-cutoff dataset. An explicitly bounded holdout window
is evaluated only once. No online model, trades, or real-data skill claims.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import fmean

from tris_market_event_intelligence.cpu_baseline import evaluate_empirical_baseline
from tris_market_event_intelligence.event_ticker_screen import EventTickerLink
from tris_market_event_intelligence.exchange_sessions import utc
from tris_market_event_intelligence.intraday_event_study import (
    IntradayNews,
    IntradayResult,
)
from tris_market_event_intelligence.walkforward_dataset import build_walkforward_dataset


@dataclass(frozen=True)
class RollingWindow:
    cutoff_at: datetime
    holdout_end_at: datetime


def evaluate_rolling_windows(
    events: Iterable[IntradayNews],
    links: Iterable[EventTickerLink],
    results: Iterable[IntradayResult],
    *,
    windows: Iterable[RollingWindow],
    embargo_minutes: int = 60,
    min_train_events: int = 3,
    min_holdout_events: int = 2,
) -> dict[str, object]:
    """Evaluate bounded disjoint holdout windows with per-fold refitting."""
    news, bindings, screened = list(events), list(links), list(results)
    folds = list(windows)
    if not 2 <= len(folds) <= 24:
        raise ValueError("Rolling evaluation requires 2..24 windows")
    if type(embargo_minutes) is not int or embargo_minutes < 60:
        raise ValueError("Embargo must be at least 60 minutes")
    if type(min_train_events) is not int or type(min_holdout_events) is not int or (
        min_train_events < 2 or min_holdout_events < 2
    ):
        raise ValueError("Independent event minimums must be at least two")
    previous_end: datetime | None = None
    normalized: list[tuple[datetime, datetime]] = []
    for window in folds:
        cutoff = utc(window.cutoff_at)
        end = utc(window.holdout_end_at)
        if end <= cutoff + timedelta(minutes=embargo_minutes):
            raise ValueError("Holdout window must extend beyond cutoff and embargo")
        if previous_end is not None and cutoff < previous_end:
            raise ValueError("Rolling test windows must be ordered and nonoverlapping")
        previous_end = end
        normalized.append((cutoff, end))

    output: list[dict[str, object]] = []
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    observed_test: set[tuple[str, int, str, int]] = set()
    for index, (cutoff, end) in enumerate(normalized, 1):
        dataset = build_walkforward_dataset(
            news, bindings, screened, cutoff=cutoff,
            embargo_minutes=embargo_minutes,
        )
        # Each incident is assigned at most once across disjoint test windows.
        # Do not apply this upper bound to train: each fold must refit on ALL
        # permissible history available before its own cutoff.
        bounded = [
            sample for sample in dataset["samples"]
            if sample["split"] == "TRAIN" or (
                sample["split"] == "HOLDOUT"
                and utc(datetime.fromisoformat(sample["available_at"])) < end
            )
        ]
        for sample in bounded:
            if sample["split"] != "HOLDOUT":
                continue
            key = (
                sample["market"], sample["event_id"], sample["symbol"],
                sample["horizon_minutes"],
            )
            if key in observed_test:
                raise ValueError("Same event-horizon repeated across test folds")
            observed_test.add(key)
        dataset["samples"] = bounded
        dataset["counts"] = {
            "TRAIN": sum(row["split"] == "TRAIN" for row in bounded),
            "HOLDOUT": sum(row["split"] == "HOLDOUT" for row in bounded),
        }
        scored = evaluate_empirical_baseline(
            dataset, min_train_events=min_train_events,
            min_holdout_events=min_holdout_events,
        )
        for group in scored["groups"]:
            if group["status"] == "RESEARCH_METRICS_ONLY":
                grouped[(group["market"], group["horizon_minutes"])].append(group)
        output.append({
            "fold": index,
            "cutoff_at": cutoff.isoformat(),
            "holdout_start_at": (
                cutoff + timedelta(minutes=embargo_minutes)
            ).isoformat(),
            "holdout_end_at_exclusive": end.isoformat(),
            "train_samples": scored["eligible_train_samples"],
            "holdout_samples": scored["eligible_holdout_samples"],
            "groups": scored["groups"],
        })

    summaries: list[dict[str, object]] = []
    for (market, horizon), rows in sorted(grouped.items()):
        summaries.append({
            "market": market, "horizon_minutes": horizon,
            "scored_folds": len(rows),
            "holdout_samples_across_folds": sum(
                row["holdout_samples"] for row in rows
            ),
            "mean_fold_empirical_mae": fmean(row["empirical_mae"] for row in rows),
            "mean_fold_zero_mae": fmean(row["zero_return_mae"] for row in rows),
            "folds_beating_zero_mae": sum(row["beats_zero_mae"] for row in rows),
            "status": (
                "MULTI_FOLD_RESEARCH_ONLY" if len(rows) >= 2
                else "INSUFFICIENT_SCORED_FOLDS"
            ),
        })
    return {
        "schema_version": "rolling_cpu_backtest_v1",
        "market_coverage": ["XKRX", "XNYS", "XNAS"],
        "embargo_minutes": embargo_minutes,
        "window_count": len(folds),
        "scored_window_count": sum(
            any(g["status"] == "RESEARCH_METRICS_ONLY" for g in f["groups"])
            for f in output
        ),
        "folds": output, "summaries": summaries,
        "status": "RESEARCH_ONLY",
        "trained_deployable_model": False,
        "real_market_validated": False,
        "predictive_accuracy_verified": False,
        "quality_verified": False,
        "note": (
            "Ex-post rolling test windows, separated by a temporal embargo. "
            "No real market provenance validation, calibration or live inference."
        ),
    }
