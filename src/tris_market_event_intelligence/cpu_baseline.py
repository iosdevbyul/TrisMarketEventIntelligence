"""CPU-only empirical mean vs zero-return baseline for historical research.

Consumes only the already purged walk-forward split. Never produces a
deployable model, live forecast, or claims real-market predictive accuracy.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from math import isfinite, sqrt
from statistics import fmean

from tris_market_event_intelligence.exchange_sessions import utc

_FEATURES = frozenset({
    "market", "symbol", "relationship", "hour_of_day", "weekday",
    "source_latency_seconds", "horizon_minutes",
})


def _validated_rows(dataset: dict[str, object]) -> tuple[list[dict], list[dict]]:
    if dataset.get("schema_version") != "walkforward_dataset_v1":
        raise ValueError("Expected walkforward_dataset_v1 research dataset")
    cutoff = utc(datetime.fromisoformat(dataset["cutoff_utc"]))
    embargo = dataset["embargo_minutes"]
    horizons = dataset["horizons_minutes"]
    if not isinstance(embargo, int) or isinstance(embargo, bool) or (
        not isinstance(horizons, list) or not horizons
        or any(type(h) is not int or h < 1 for h in horizons)
        or embargo < max(horizons)
    ):
        raise ValueError("Invalid research embargo or horizon contract")
    rows = dataset.get("samples")
    if not isinstance(rows, list):
        raise TypeError("Samples must be a list")
    train: list[dict] = []
    holdout: list[dict] = []
    seen: set[tuple[str, int, str, int]] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("features"), dict):
            raise TypeError("Malformed research sample")
        features = row["features"]
        if set(features) != _FEATURES:
            raise ValueError("Features must match strict point-in-time allowlist")
        target = row.get("target_excess_return")
        if type(target) not in (float, int) or not isfinite(target):
            raise ValueError("Historical target must be a finite number")
        market, horizon = features["market"], features["horizon_minutes"]
        if market not in {"XKRX", "XNYS", "XNAS"} or (
            type(horizon) is not int or horizon not in horizons
            or row.get("market") != market or row.get("horizon_minutes") != horizon
            or not isinstance(features["symbol"], str)
            or row.get("symbol") != features["symbol"]
        ):
            raise ValueError("Market/horizon/symbol mismatch")
        available = utc(datetime.fromisoformat(row["available_at"]))
        exit_at = utc(datetime.fromisoformat(row["label_exit_at"]))
        if exit_at <= available or exit_at > available + timedelta(minutes=horizon + 1):
            raise ValueError("Invalid target observation interval")
        event_id = row.get("event_id")
        if type(event_id) is not int or event_id < 1:
            raise ValueError("Invalid event ID")
        key = market, event_id, features["symbol"], horizon
        if key in seen:
            raise ValueError("Duplicate event-horizon research sample")
        seen.add(key)
        split = row.get("split")
        if split == "TRAIN" and exit_at < cutoff - timedelta(minutes=embargo):
            train.append(row)
        elif split == "HOLDOUT" and available >= cutoff + timedelta(minutes=embargo):
            holdout.append(row)
        else:
            raise ValueError("Sample violates purged chronological split")
    train_events = {(r["market"], r["event_id"], r["symbol"]) for r in train}
    holdout_events = {(r["market"], r["event_id"], r["symbol"]) for r in holdout}
    if train_events & holdout_events:
        raise ValueError("Same event observed in train and holdout")
    return train, holdout


def evaluate_empirical_baseline(
    dataset: dict[str, object], *,
    min_train_events: int = 3, min_holdout_events: int = 2,
) -> dict[str, object]:
    """Compare mean historical excess-return baseline with a zero predictor.

    Group by market+horizon only, never select or tune on holdout. Both
    baselines are intentionally weak and do not establish predictive skill.
    """
    if min_train_events < 2 or min_holdout_events < 2:
        raise ValueError("At least two independent events required per partition")
    train, holdout = _validated_rows(dataset)
    groups: dict[tuple[str, int], dict[str, list[dict]]] = defaultdict(
        lambda: {"TRAIN": [], "HOLDOUT": []},
    )
    for row in train:
        groups[(row["market"], row["horizon_minutes"])]["TRAIN"].append(row)
    for row in holdout:
        groups[(row["market"], row["horizon_minutes"])]["HOLDOUT"].append(row)
    scores: list[dict[str, object]] = []
    for (market, horizon), partitions in sorted(groups.items()):
        train_rows, test_rows = partitions["TRAIN"], partitions["HOLDOUT"]
        train_events = {(x["event_id"], x["symbol"]) for x in train_rows}
        test_events = {(x["event_id"], x["symbol"]) for x in test_rows}
        base = {
            "market": market, "horizon_minutes": horizon,
            "train_samples": len(train_rows), "train_distinct_events": len(train_events),
            "holdout_samples": len(test_rows),
            "holdout_distinct_events": len(test_events),
        }
        if len(train_events) < min_train_events or len(test_events) < min_holdout_events:
            scores.append({**base, "status": "INSUFFICIENT_INDEPENDENT_EVENTS"})
            continue
        prediction = fmean(float(x["target_excess_return"]) for x in train_rows)
        actual = [float(x["target_excess_return"]) for x in test_rows]
        empirical_mae = fmean(abs(x - prediction) for x in actual)
        zero_mae = fmean(abs(x) for x in actual)
        empirical_rmse = sqrt(fmean((x - prediction) ** 2 for x in actual))
        zero_rmse = sqrt(fmean(x ** 2 for x in actual))
        # A 0% return is a neutral forecast, not a positive directional call.
        direction_accuracy = (
            fmean(float((prediction > 0) == (x > 0)) for x in actual)
            if prediction != 0 else None
        )
        scores.append({
            **base, "status": "RESEARCH_METRICS_ONLY",
            "train_mean_excess_return": prediction,
            "empirical_mae": empirical_mae, "zero_return_mae": zero_mae,
            "empirical_rmse": empirical_rmse, "zero_return_rmse": zero_rmse,
            "direction_accuracy": direction_accuracy,
            "beats_zero_mae": empirical_mae < zero_mae,
            "beats_zero_rmse": empirical_rmse < zero_rmse,
        })
    return {
        "schema_version": "cpu_baseline_evaluation_v1",
        "dataset_schema_version": "walkforward_dataset_v1",
        "cutoff_utc": dataset["cutoff_utc"],
        "eligible_train_samples": len(train),
        "eligible_holdout_samples": len(holdout),
        "groups": scores,
        "status": "RESEARCH_ONLY",
        "trained_deployable_model": False,
        "real_market_validated": False,
        "predictive_accuracy_verified": False,
        "quality_verified": False,
        "note": (
            "CPU-only retrospective empirical mean vs zero-return benchmark; "
            "synthetic/offline research cannot justify live predictions. "
            "Independent real-data provenance and rolling out-of-period windows remain required."
        ),
    }
