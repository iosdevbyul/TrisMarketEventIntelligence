"""PIT-aligned observational stock outcomes built from old source-pinned inputs."""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from test_historical_shock_research import _setup

from tris_market_event_intelligence.historical_price_outcome_cohort_cli import (
    _private_write,
    build_news_price_outcome_cohort,
)


def _build(tmp_path: Path) -> dict:
    stock, bench, macro, news = _setup(tmp_path)
    return build_news_price_outcome_cohort(
        stock_csv=stock,
        benchmark_csv=bench,
        news_cohort=news,
        macro_csv=macro,
        ticker="NVDA",
        benchmark="SPY",
    )


def test_pit_features_and_price_truth_are_separated(tmp_path: Path) -> None:
    report = _build(tmp_path)
    assert report["observationRows"] == 44
    assert report["trainingReady"] is False
    assert report["forwardPredictionsSealed"] is False
    assert report["rowsWithoutPreCutoffNews"] > 0
    first = report["records"][0]
    assert first["features"]["lastKnownStockSession"] == "2026-10-01"
    assert first["decisionSession"] == "2026-10-02"
    assert first["features"]["news"] == []
    assert first["outcome"]["horizonsTradingSessions"]["1"]["targetSession"] == (
        "2026-10-02"
    )
    assert first["outcome"]["horizonsTradingSessions"]["3"]["targetSession"] == (
        "2026-10-06"
    )
    assert first["outcome"]["notARealTimeTradableEntry"] is True
    assert first["trainingEligible"] is False
    assert "stockCloseToCloseReturn" not in json.dumps(first["features"])
    assert "excessReturn" not in json.dumps(first["features"])
    assert len(report["stockPriceSha256"]) == 64
    assert len(report["newsCohortSha256"]) == 64
    # No revised November CPI value may enter the October feature set.
    early = next(x for x in report["records"] if x["decisionSession"] == "2026-10-21")
    assert early["features"]["macroPointInTime"][0]["value"] == 3.0


def test_every_model_news_row_was_observed_by_cutoff(tmp_path: Path) -> None:
    report = _build(tmp_path)
    for row in report["records"]:
        cutoff = datetime.fromisoformat(row["asOf"]).astimezone(UTC)
        for article in row["features"]["news"]:
            assert datetime.fromisoformat(article["firstObservedAt"]) <= cutoff
            assert len(article["captureSha256"]) == 64
            assert article["issuerAssociationVerified"] is False
        assert row["outcome"]["labelAvailability"] == "AFTER_TARGET_SESSION_CLOSE"


def test_future_news_is_excluded_instead_of_backdated(tmp_path: Path) -> None:
    stock, bench, _, news = _setup(tmp_path)
    doc = json.loads(news.read_text())
    # Move original article evidence + its source as-of forward: violates
    # source cohort cutoff, and must not be accepted as older training news.
    doc["items"][0]["firstObservedAt"] = (
        datetime.fromisoformat(doc["asOf"]) + timedelta(days=1)
    ).isoformat()
    news.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="cohort cutoff"):
        build_news_price_outcome_cohort(
            stock_csv=stock, benchmark_csv=bench, news_cohort=news,
            ticker="NVDA", benchmark="SPY",
        )


def test_original_sources_unchanged_and_private_output(tmp_path: Path) -> None:
    stock, bench, macro, news = _setup(tmp_path)
    originals = [p.read_bytes() for p in (stock, bench, macro, news)]
    report = build_news_price_outcome_cohort(
        stock_csv=stock, benchmark_csv=bench, news_cohort=news,
        macro_csv=macro, ticker="NVDA", benchmark="SPY",
    )
    assert [p.read_bytes() for p in (stock, bench, macro, news)] == originals
    target = tmp_path / "outcomes.json"
    _private_write(target, report)
    assert target.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="new"):
        _private_write(target, report)


def test_price_truth_not_training_gold(tmp_path: Path) -> None:
    report = _build(tmp_path)
    assert report["humanGoldCreated"] is False
    assert report["sourceIssuerAssociationVerified"] is False
    assert report["modelPromotionAllowed"] is False
    assert all(row["modelPredictionGenerated"] is False
               for row in report["records"])
