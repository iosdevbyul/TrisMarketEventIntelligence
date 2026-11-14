"""Synthetic-only CPU research baseline safety and holdout tests."""
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tris_market_event_intelligence.cpu_baseline import evaluate_empirical_baseline
from tris_market_event_intelligence.cpu_baseline_cli import evaluate_report

CUTOFF = datetime(2026, 11, 17, 12, tzinfo=UTC)
HORIZONS = [1, 5, 15, 30, 60]


def sample(
    event_id: int, *, split: str, market: str = "XNAS", horizon: int = 5,
    target: float = 0.01, minutes_from_cutoff: int = -200,
) -> dict:
    available = CUTOFF + timedelta(minutes=minutes_from_cutoff)
    exit_at = available + timedelta(minutes=horizon + 1)
    return {
        "event_id": event_id,
        "market": market,
        "symbol": "NVDA" if market == "XNAS" else "005930",
        "horizon_minutes": horizon,
        "available_at": available.isoformat(),
        "label_exit_at": exit_at.isoformat(),
        "split": split,
        "features": {
            "market": market,
            "symbol": "NVDA" if market == "XNAS" else "005930",
            "relationship": "DIRECT",
            "hour_of_day": available.hour,
            "weekday": available.weekday(),
            "source_latency_seconds": 10.0,
            "horizon_minutes": horizon,
        },
        "target_excess_return": target,
    }


def dataset(rows: list[dict]) -> dict:
    return {
        "schema_version": "walkforward_dataset_v1",
        "cutoff_utc": CUTOFF.isoformat(),
        "embargo_minutes": 60,
        "horizons_minutes": HORIZONS,
        "samples": rows,
        "training_dataset_eligible": False,
        "point_in_time_verified": False,
    }


def test_holdout_score_uses_training_mean_not_holdout_average() -> None:
    rows = [
        sample(1, split="TRAIN", target=0.01),
        sample(2, split="TRAIN", target=0.03),
        sample(3, split="TRAIN", target=0.02),
        sample(4, split="HOLDOUT", target=0.02, minutes_from_cutoff=120),
        sample(5, split="HOLDOUT", target=-0.01, minutes_from_cutoff=180),
    ]
    result = evaluate_empirical_baseline(dataset(rows))
    group = result["groups"][0]
    assert group["status"] == "RESEARCH_METRICS_ONLY"
    assert group["train_mean_excess_return"] == pytest.approx(0.02)
    assert group["empirical_mae"] == pytest.approx(0.015)
    assert group["zero_return_mae"] == pytest.approx(0.015)
    assert group["direction_accuracy"] == pytest.approx(0.5)
    assert group["beats_zero_mae"] is False
    assert result["real_market_validated"] is False
    assert result["trained_deployable_model"] is False
    assert result["predictive_accuracy_verified"] is False


def test_single_event_cannot_produce_a_research_score() -> None:
    rows = [
        sample(1, split="TRAIN"),
        sample(2, split="HOLDOUT", minutes_from_cutoff=120),
    ]
    group = evaluate_empirical_baseline(dataset(rows))["groups"][0]
    assert group["status"] == "INSUFFICIENT_INDEPENDENT_EVENTS"
    assert "empirical_mae" not in group


def test_groups_do_not_mix_us_and_kr_or_minute_horizons() -> None:
    rows = []
    for market in ("XNAS", "XKRX"):
        for horizon in (1, 15):
            for event_id in (1, 2, 3):
                rows.append(sample(
                    event_id, split="TRAIN", market=market, horizon=horizon,
                ))
            for event_id in (4, 5):
                rows.append(sample(
                    event_id, split="HOLDOUT", market=market, horizon=horizon,
                    minutes_from_cutoff=120,
                ))
    scores = evaluate_empirical_baseline(dataset(rows))["groups"]
    assert len(scores) == 4
    assert all(s["train_distinct_events"] == 3 for s in scores)
    assert all(s["holdout_distinct_events"] == 2 for s in scores)


def test_no_holdout_or_no_training_never_promises_accuracy() -> None:
    rows = [
        sample(i, split="TRAIN") for i in (1, 2, 3)
    ]
    result = evaluate_empirical_baseline(dataset(rows))
    assert result["groups"][0]["status"] == "INSUFFICIENT_INDEPENDENT_EVENTS"
    assert result["predictive_accuracy_verified"] is False
    assert evaluate_empirical_baseline(dataset([]))["groups"] == []


def test_change_of_holdout_label_cannot_change_training_prediction() -> None:
    base = [
        sample(1, split="TRAIN", target=0.01),
        sample(2, split="TRAIN", target=0.02),
        sample(3, split="TRAIN", target=0.03),
        sample(4, split="HOLDOUT", target=0.01, minutes_from_cutoff=120),
        sample(5, split="HOLDOUT", target=0.01, minutes_from_cutoff=180),
    ]
    first = evaluate_empirical_baseline(dataset(base))["groups"][0]
    changed = [dict(row) for row in base]
    changed[3]["target_excess_return"] = -0.3
    second = evaluate_empirical_baseline(dataset(changed))["groups"][0]
    assert first["train_mean_excess_return"] == second["train_mean_excess_return"]
    assert first["empirical_mae"] != second["empirical_mae"]


def test_rejects_leaked_future_feature_and_duplicate_label() -> None:
    row = sample(1, split="TRAIN")
    leaked = {**row, "features": {**row["features"], "future_close": 100.0}}
    with pytest.raises(ValueError, match="allowlist"):
        evaluate_empirical_baseline(dataset([leaked]))
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_empirical_baseline(dataset([row, row]))


@pytest.mark.parametrize("target", [float("nan"), float("inf"), True, "0.03"])
def test_rejects_non_numeric_or_nonfinite_targets(target: object) -> None:
    row = sample(1, split="TRAIN")
    row["target_excess_return"] = target
    with pytest.raises(ValueError, match="finite"):
        evaluate_empirical_baseline(dataset([row]))


def test_rejects_wrong_split_time_and_recycled_event() -> None:
    bad = sample(2, split="TRAIN", minutes_from_cutoff=120)
    with pytest.raises(ValueError, match="purged chronological"):
        evaluate_empirical_baseline(dataset([bad]))
    row = sample(1, split="TRAIN")
    reused = sample(1, split="HOLDOUT", minutes_from_cutoff=120)
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_empirical_baseline(dataset([row, reused]))


def test_cli_reads_existing_intraday_report_without_modifying_it(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    body = {"walkforward_dataset": dataset([
        sample(1, split="TRAIN"),
        sample(2, split="TRAIN"),
        sample(3, split="TRAIN"),
        sample(4, split="HOLDOUT", minutes_from_cutoff=120),
        sample(5, split="HOLDOUT", minutes_from_cutoff=180),
    ])}
    path.write_text(json.dumps(body), encoding="utf-8")
    before = path.read_bytes()
    result = evaluate_report(path)
    assert result["groups"][0]["status"] == "RESEARCH_METRICS_ONLY"
    assert path.read_bytes() == before


def test_invalid_embargo_and_wrong_schema_rejected() -> None:
    with pytest.raises(ValueError, match="Expected walkforward"):
        evaluate_empirical_baseline({})
    data = dataset([])
    data["embargo_minutes"] = 10
    with pytest.raises(ValueError, match="embargo"):
        evaluate_empirical_baseline(data)
