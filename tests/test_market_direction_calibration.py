"""Training-journal-only calibration tests and temporal holdout isolation."""
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from tris_market_event_intelligence.market_direction_calibration import (
    calibrate_direction,
    fit_price_calibrator,
)


def journal(path, *, n: int = 10, wrong_future: bool = False) -> None:
    release = datetime(2026, 11, 16, tzinfo=UTC)
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE steps(seq INTEGER, observed_at TEXT, phase TEXT, "
            "prediction_json TEXT, evaluation_json TEXT)"
        )
        for i in range(n):
            published = release - timedelta(days=40 - i)
            outcome = (published + timedelta(days=1)).isoformat()
            if i == n - 1 and wrong_future:
                outcome = (release + timedelta(hours=2)).isoformat()
            pred = {"impacts": [
                {"targetType": "MARKET", "target": "BROAD_MARKET",
                 "priceProxy": "069500.KS", "direction": "NEUTRAL"},
            ]}
            evaluation = {"impacts": [
                {"targetType": "MARKET", "target": "BROAD_MARKET",
                 "priceProxy": "069500.KS", "direction": "NEUTRAL",
                 "actualDirection": "NEGATIVE", "status": "WRONG",
                 "outcomeAt": outcome},
            ]}
            db.execute("INSERT INTO steps VALUES (?, ?, ?, ?, ?)", (
                i, published.isoformat(), "COMPLETE",
                json.dumps(pred), json.dumps(evaluation),
            ))


def test_training_calibrator_frozen_before_first_holdout_and_market_specific(tmp_path):
    path = tmp_path / "train.sqlite3"
    journal(path)
    earliest = datetime(2026, 11, 16, tzinfo=UTC)
    trained = fit_price_calibrator(path, earliest, minimum_support=8)
    assert trained["status"] == "TRAINED"
    assert trained["ruleCount"] == 1
    assert trained["trainingTargetObservations"] == 10
    result = calibrate_direction(trained, "069500.KS", "NEUTRAL")
    assert result["status"] == "AVAILABLE"
    assert result["direction"] == "NEGATIVE"
    assert result["support"] == 10
    # A KRX-trained price shortcut never becomes a US-equity rule.
    assert calibrate_direction(trained, "NVDA", "NEUTRAL")["status"] == "ABSTAIN"
    assert trained["causalLearningVerified"] is False


def test_post_holdout_outcome_cannot_enter_training(tmp_path):
    path = tmp_path / "train.sqlite3"
    journal(path, wrong_future=True)
    fitted = fit_price_calibrator(
        path, datetime(2026, 11, 16, tzinfo=UTC),
        minimum_support=8,
    )
    assert fitted["skippedUnmatured"] == 1
    assert fitted["trainingTargetObservations"] == 9
    assert fitted["rules"]["KR:NEUTRAL"]["support"] == 9


def test_no_supported_historical_rule_is_explicit_abstention(tmp_path):
    path = tmp_path / "train.sqlite3"
    journal(path, n=2)
    fitted = fit_price_calibrator(
        path, datetime(2026, 11, 16, tzinfo=UTC),
        minimum_support=8,
    )
    assert fitted["status"] == "NO_SUPPORTED_RULES"
    assert calibrate_direction(fitted, "069500.KS", "NEUTRAL")["status"] == "ABSTAIN"


def test_train_publication_overlap_rejected_even_if_outcome_missing(tmp_path):
    path = tmp_path / "train.sqlite3"
    journal(path, n=10)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE steps SET observed_at=? WHERE seq=0", (
            "2026-11-16T00:00:00+00:00",
        ))
    with pytest.raises(ValueError, match="overlaps"):
        fit_price_calibrator(path, datetime(2026, 11, 16, tzinfo=UTC))


def test_near_even_historical_votes_do_not_produce_false_learning(tmp_path):
    path = tmp_path / "near_even.sqlite3"
    journal(path, n=74)
    with sqlite3.connect(path) as db:
        for i in range(74):
            released = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=i)
            observed = (
                "POSITIVE" if i < 35 else
                "NEGATIVE" if i < 71 else "NEUTRAL"
            )
            evaluation = {
                "impacts": [{
                    "targetType": "MARKET",
                    "target": "BROAD_MARKET",
                    "priceProxy": "069500.KS",
                    "direction": "NEUTRAL",
                    "actualDirection": observed,
                    "status": "CORRECT" if observed == "NEUTRAL" else "WRONG",
                    "outcomeAt": (released + timedelta(days=1)).isoformat(),
                }],
            }
            db.execute(
                "UPDATE steps SET observed_at=?, evaluation_json=? WHERE seq=?",
                (released.isoformat(), json.dumps(evaluation), i),
            )
    trained = fit_price_calibrator(
        path, datetime(2026, 11, 16, tzinfo=UTC), minimum_support=8,
    )
    assert trained["ruleCount"] == 0
    assert trained["status"] == "NO_SUPPORTED_RULES"
    rejected = trained["rejectedBuckets"]["KR:NEUTRAL"]
    assert rejected["reasonCode"] == "NO_DOMINANT_HISTORICAL_DIRECTION"
    assert rejected["distribution"] == {
        "NEGATIVE": 36, "NEUTRAL": 3, "POSITIVE": 35,
    }
    assert rejected["leaderMargin"] < trained["minimumLeaderMargin"]
    assert calibrate_direction(trained, "069500.KS", "NEUTRAL")["status"] == "ABSTAIN"
