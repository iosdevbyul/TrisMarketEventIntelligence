"""Explicitly same-source empirical comparison for real news price rechecks.

Fixtures are synthetic. No test claims realized market predictive accuracy.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tris_market_event_intelligence.same_cohort_market_recheck_cli import (
    compare_same_cohort,
)

SHA = "a" * 64


def fixture(tmp: Path) -> tuple[Path, Path]:
    mappings = tmp / "mappings.csv"
    with mappings.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["target", "market", "symbol", "benchmark", "verified_at"])
        writer.writerow(["Nvidia", "XNAS", "NVDA", "SPY", "2026-11-24T00:00:00+00:00"])
    prices = tmp / "adjusted.csv"
    with prices.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["symbol", "date", "adjusted_close", "source"])
        writer.writerows([
            ["NVDA", "2026-11-20", "200", "synthetic"],
            ["SPY", "2026-11-20", "400", "synthetic"],
        ])
    return mappings, prices


def impact(article: int, *, status: str, verdict: str | None = None) -> dict:
    return {
        "articleId": article,
        "impactIndex": 0,
        "predictionSourceSha256": SHA,
        "target": "Nvidia",
        "prediction": "UP",
        "predictionSealedBeforeOutcome": False,
        "publishedAt": "2026-11-23T00:08:00+00:00",
        "status": status,
        "mapping": {"symbol": "NVDA", "benchmark": "SPY"},
        "horizons": (
            [
                {
                    "sessions": n,
                    "verdict": verdict,
                    "dataStatus": ("COMPLETED" if verdict else "NO_ENTRY_SESSION"),
                    "exitDay": None,
                }
                for n in (1, 5)
            ]
            if verdict is not None else []
        ),
    }


def report(items: list[dict]) -> dict:
    return {"impactsInspected": len(items), "items": items}


def test_13_new_predictions_cannot_fake_a_scored_improvement(
    tmp_path: Path,
) -> None:
    mapping, prices = fixture(tmp_path)
    old = report([impact(20975, status="UNMAPPED")])
    new = report([
        impact(20975, status="NO_ENTRY_SESSION"),
        *[impact(30000 + i, status="UNMAPPED") for i in range(13)],
    ])
    result = compare_same_cohort(old, new, mapping, prices)
    assert result["previousImpactCount"] == 1
    assert result["currentImpactCount"] == 14
    assert result["commonIdenticalSourceImpacts"] == 1
    assert result["newSourceImpacts"] == 13
    assert result["removedOrChangedSourceImpacts"] == 0
    assert result["sameSourceHorizonScores"]["1d"]["newlyScored"] == 0
    assert result["sameSourceHorizonScores"]["5d"]["newlyScored"] == 0
    assert result["notYetMaturedCount"] == 1
    pending = result["notYetMatured"][0]
    assert pending["target"] == "Nvidia"
    assert pending["lastInstrumentDay"] == "2026-11-20"
    assert pending["lastBenchmarkDay"] == "2026-11-20"
    assert pending["reason"] == "NO_POST_PUBLICATION_CLOSE_IN_CURRENT_FILE"
    assert result["trainingEligible"] is False


def test_only_identical_original_source_impacts_can_improve_score(
    tmp_path: Path,
) -> None:
    mapping, prices = fixture(tmp_path)
    before = report([impact(20975, status="NO_ENTRY_SESSION")])
    after = report([impact(20975, status="COMPLETED", verdict="CORRECT")])
    result = compare_same_cohort(before, after, mapping, prices)
    assert result["cohortsIdentical"] is True
    assert result["sameSourceHorizonScores"]["1d"] == {
        "before": 0, "after": 1, "newlyScored": 1, "lostScores": 0,
    }
    assert result["sameSourceHorizonScores"]["5d"]["newlyScored"] == 1


def test_changed_source_digest_cannot_be_compared_to_previous_forecast(
    tmp_path: Path,
) -> None:
    mapping, prices = fixture(tmp_path)
    old = impact(20975, status="NO_ENTRY_SESSION")
    new = impact(20975, status="COMPLETED", verdict="CORRECT")
    new["predictionSourceSha256"] = "b" * 64
    result = compare_same_cohort(report([old]), report([new]), mapping, prices)
    assert result["commonIdenticalSourceImpacts"] == 0
    assert result["newSourceImpacts"] == 1
    assert result["removedOrChangedSourceImpacts"] == 1
    assert result["sameSourceHorizonScores"]["1d"]["newlyScored"] == 0


def test_reject_forged_prospective_impact_and_duplicate_identity(
    tmp_path: Path,
) -> None:
    mapping, prices = fixture(tmp_path)
    valid = impact(20975, status="UNMAPPED")
    forged = {**valid, "predictionSealedBeforeOutcome": True}
    with pytest.raises(ValueError, match="unsealed"):
        compare_same_cohort(report([valid]), report([forged]), mapping, prices)
    with pytest.raises(ValueError, match="Duplicate"):
        compare_same_cohort(report([valid]), report([valid, valid]), mapping, prices)


def test_empty_common_cohort_has_no_artificial_accuracy(tmp_path: Path) -> None:
    mapping, prices = fixture(tmp_path)
    result = compare_same_cohort(
        report([impact(1, status="UNMAPPED")]),
        report([impact(2, status="UNMAPPED")]),
        mapping, prices,
    )
    assert result["commonIdenticalSourceImpacts"] == 0
    assert result["sameSourceHorizonScores"]["1d"]["after"] == 0
    assert result["sameSourceHorizonScores"]["5d"]["newlyScored"] == 0
    assert result["prospectivePredictionAccuracy"] is None
