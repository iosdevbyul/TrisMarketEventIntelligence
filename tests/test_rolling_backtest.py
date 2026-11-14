"""Rolling historical evaluation isolates disjoint future windows."""
from datetime import UTC, datetime, timedelta

import pytest

from tris_market_event_intelligence.event_ticker_screen import EventTickerLink
from tris_market_event_intelligence.intraday_event_study import (
    IntradayLabel,
    IntradayNews,
    IntradayResult,
)
from tris_market_event_intelligence.rolling_backtest import (
    RollingWindow,
    evaluate_rolling_windows,
)


def incident(event_id: int, day: int, *, market: str = "XNAS") -> tuple:
    stamp = datetime(2026, 11, day, 10, 15, tzinfo=UTC)
    entry = stamp + timedelta(minutes=1)
    symbol = "NVDA" if market == "XNAS" else "005930"
    e = IntradayNews(event_id, market, symbol, "SPY", stamp, stamp, stamp)
    link = EventTickerLink(
        event_id, market, symbol, "SPY", f"https://example.org/{event_id}",
        stamp, "DIRECT",
    )
    labels = tuple(
        IntradayLabel(
            h, (entry + timedelta(minutes=h)).isoformat(), 0.02, 0.01,
            0.01 if event_id % 2 else -0.01, "COMPLETED",
        ) for h in (1, 5, 15, 30, 60)
    )
    r = IntradayResult(
        event_id, market, symbol, "SPY", entry.isoformat(), "COMPLETED",
        0, labels,
    )
    return e, link, r


def windows() -> list[RollingWindow]:
    return [
        RollingWindow(
            datetime(2026, 11, 4, tzinfo=UTC),
            datetime(2026, 11, 7, tzinfo=UTC),
        ),
        RollingWindow(
            datetime(2026, 11, 8, tzinfo=UTC),
            datetime(2026, 11, 12, tzinfo=UTC),
        ),
    ]


def evaluate(items: list[tuple], folds: list[RollingWindow] | None = None) -> dict:
    return evaluate_rolling_windows(
        [x[0] for x in items], [x[1] for x in items],
        [x[2] for x in items], windows=folds or windows(),
    )


def test_disjoint_two_fold_evaluation_retrains_on_prior_known_history() -> None:
    # 3 events prior to fold 1, two in the first test period,
    # two in the second test period.
    data = [incident(i, d) for i, d in enumerate(
        [1, 2, 3, 5, 6, 9, 10], start=1,
    )]
    result = evaluate(data)
    assert result["window_count"] == 2
    assert result["scored_window_count"] == 2
    assert result["folds"][0]["train_samples"] == 15
    assert result["folds"][0]["holdout_samples"] == 10
    assert result["folds"][1]["train_samples"] == 25
    assert result["folds"][1]["holdout_samples"] == 10
    assert len(result["summaries"]) == 5
    assert all(g["scored_folds"] == 2 for g in result["summaries"])
    assert result["real_market_validated"] is False
    assert result["trained_deployable_model"] is False
    assert result["predictive_accuracy_verified"] is False


def test_perturb_second_test_period_does_not_change_earlier_fold_scores() -> None:
    rows = [incident(i, d) for i, d in enumerate(
        [1, 2, 3, 5, 6, 9, 10], start=1,
    )]
    first = evaluate(rows)
    event, link, result = rows[-1]
    replaced = tuple(
        IntradayLabel(
            label.horizon_minutes, label.exit_at, label.stock_return,
            label.benchmark_return, 0.8, label.status,
        ) for label in result.labels
    )
    rows[-1] = (event, link, IntradayResult(
        result.event_id, result.market, result.symbol,
        result.benchmark_symbol, result.entry_at, result.status,
        result.latency_seconds, replaced,
    ))
    changed = evaluate(rows)
    assert first["folds"][0] == changed["folds"][0]
    assert first["folds"][1]["groups"] != changed["folds"][1]["groups"]


def test_empty_or_sparse_windows_report_no_earned_accuracy() -> None:
    result = evaluate([incident(1, 2), incident(2, 5)])
    assert result["scored_window_count"] == 0
    assert result["summaries"] == []
    assert all(
        group["status"] == "INSUFFICIENT_INDEPENDENT_EVENTS"
        for fold in result["folds"] for group in fold["groups"]
    )


def test_windows_must_be_bounded_ordered_and_nonoverlapping() -> None:
    w = windows()
    with pytest.raises(ValueError, match="nonoverlapping"):
        evaluate([], [w[1], w[0]])
    with pytest.raises(ValueError, match="nonoverlapping"):
        evaluate([], [w[0], RollingWindow(
            datetime(2026, 11, 6, tzinfo=UTC), w[1].holdout_end_at,
        )])
    with pytest.raises(ValueError, match="beyond cutoff"):
        evaluate([], [RollingWindow(
            datetime(2026, 11, 4, tzinfo=UTC),
            datetime(2026, 11, 4, 0, 30, tzinfo=UTC),
        ), w[1]])
    with pytest.raises(ValueError, match="2..24"):
        evaluate_rolling_windows([], [], [], windows=[w[0]])


def test_naive_window_and_invalid_embargo_rejected() -> None:
    with pytest.raises(ValueError, match="timezone"):
        evaluate([], [RollingWindow(
            datetime(2026, 11, 4), datetime(2026, 11, 7, tzinfo=UTC),
        ), windows()[1]])
    with pytest.raises(ValueError, match="at least 60"):
        evaluate_rolling_windows([], [], [], windows=windows(), embargo_minutes=10)


def test_outside_holdout_end_never_enters_earlier_fold() -> None:
    data = [incident(i, d) for i, d in enumerate(
        [1, 2, 3, 7, 9, 10], start=1,
    )]
    result = evaluate(data)
    assert result["folds"][0]["holdout_samples"] == 0
    assert result["folds"][1]["holdout_samples"] == 10
    assert result["scored_window_count"] == 1


def test_cross_market_groups_are_independent() -> None:
    data = []
    for m in ("XNAS", "XKRX"):
        data += [incident(i, d, market=m) for i, d in enumerate(
            [1, 2, 3, 5, 6, 9, 10], start=1,
        )]
    report = evaluate(data)
    assert len(report["summaries"]) == 10
    assert {x["market"] for x in report["summaries"]} == {"XNAS", "XKRX"}
