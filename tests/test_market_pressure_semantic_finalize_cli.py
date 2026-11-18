"""Require explicit evidence-based human opinions before semantic finalization."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_market_pressure_replay_cli import _trace_fixture

from tris_market_event_intelligence.market_pressure_semantic_finalize_cli import (
    finalize_review,
)
from tris_market_event_intelligence.market_pressure_semantic_review_cli import (
    semantic_review,
)

_REQUIRED = (
    "sourceFactSupported", "targetExposurePlausible",
    "transmissionMechanismPlausible", "directionSupported",
    "strengthSupported", "countervailingFactorsConsidered",
)


def _completed_review(tmp_path: Path) -> dict:
    trace = tmp_path / "input.jsonl"
    _trace_fixture(trace)
    review = semantic_review(trace)
    for item in review["reviewItems"]:
        item["humanReview"].update({
            **{key: True for key in _REQUIRED},
            "humanDecision": "SUPPORTED_WITH_LIMITATIONS",
            "reason": (
                "The cited observation and conditional transmission were "
                "separately reviewed, including uncertainty and opposing risks."
            ),
        })
    return review


def test_review_requires_explicit_judgments_and_reason(tmp_path: Path) -> None:
    trace = tmp_path / "input.jsonl"
    _trace_fixture(trace)
    pending = semantic_review(trace)
    with pytest.raises(ValueError, match="ALL_HUMAN"):
        finalize_review(pending, reviewer="Researcher")

    done = _completed_review(tmp_path)
    done["reviewItems"][0]["humanReview"]["strengthSupported"] = None
    with pytest.raises(ValueError, match="ALL_HUMAN"):
        finalize_review(done, reviewer="Researcher")


def test_review_never_claims_independent_accuracy(tmp_path: Path) -> None:
    review = _completed_review(tmp_path)
    review["reviewItems"][0]["humanReview"]["strengthSupported"] = False
    with pytest.raises(ValueError, match="HUMAN_SUPPORTED_BUT_CRITERIA_FAILED"):
        finalize_review(review, reviewer="Researcher")
    review["reviewItems"][0]["humanReview"]["humanDecision"] = "NOT_SUPPORTED"
    result = finalize_review(review, reviewer="Researcher")
    assert result["semanticAccuracyVerifiedByIndependentStudy"] is False
    assert result["judgmentCounts"]["NOT_SUPPORTED"] == 1
    assert result["judgmentCounts"]["SUPPORTED_WITH_LIMITATIONS"] == 1
    assert result["dbWrites"] == 0
    assert result["modelCalls"] == 0
    assert len(result["sourceTraceSha256"]) == 64
    assert len(result["reviewFileSha256"]) == 64
    assert result["items"][0]["proposalIndex"] == 0
    assert result["items"][1]["proposalIndex"] == 1
    assert "reviewItems" in json.dumps(review)
