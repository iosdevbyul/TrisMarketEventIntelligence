"""Prevent structural CI success from hiding worse semantic outcomes."""
from __future__ import annotations

import copy

import pytest

from tris_market_event_intelligence.historical_semantic_regression_cli import (
    compare_scores,
)


def score(*, decision: float = 0.875, critical: int = 5) -> dict:
    return {
        "schemaVersion": "historical_semantic_score_v2",
        "referenceStatus": "PROVISIONAL_CURATOR_REFERENCE",
        "evaluationSplit": "DEVELOPMENT_REUSED_FOR_MODEL_TUNING",
        "rates": {
            "sourceCoverage": 1.0,
            "requiredEvidenceRecall": 1.0,
            "decisionAccuracy": decision,
            "evidenceStatusAccuracy": 0.875,
            "completionSafety": 1.0,
            "primaryEvidenceAccuracy": 0.75,
            "directRecordAccuracy": 1.0,
            "indirectProgressAccuracy": 1.0,
            "criticalErrors": critical,
            "modelCompletionClaimRate": 0.125,
        },
        "caseResults": [
            {"id": "a", "status": "PASS", "errors": []},
            {
                "id": "b", "status": "REVIEW_REQUIRED",
                "errors": ["REGULATORY_DECISION_MISMATCH"],
            },
        ],
    }


def test_detects_regressions_and_new_failures() -> None:
    baseline = score()
    candidate = score(decision=0.5, critical=9)
    candidate["caseResults"][0] = {
        "id": "a", "status": "REVIEW_REQUIRED",
        "errors": ["REQUESTED_EVIDENCE_STATUS_MISMATCH"],
    }
    result = compare_scores(baseline, candidate)
    assert result["status"] == "REGRESSION_DETECTED"
    assert result["regressedMetrics"] == ["decisionAccuracy", "criticalErrors"]
    assert result["newlyFailingCases"] == ["a"]
    assert result["trustVerified"] is False
    assert result["modelCalls"] == 0


def test_no_measured_regression_does_not_establish_trust() -> None:
    baseline = score(decision=0.5, critical=9)
    candidate = score(decision=0.875, critical=5)
    result = compare_scores(baseline, candidate)
    assert result["status"] == "NO_MEASURED_REGRESSION"
    assert "decisionAccuracy" in result["improvedMetrics"]
    assert result["trustVerified"] is False


def test_rejects_mismatched_score_populations() -> None:
    baseline = score()
    for mutation, error in (
        (lambda x: x.update(schemaVersion="historical_semantic_score_v1"),
         "SCORE_SCHEMA_MISMATCH"),
        (lambda x: x.update(evaluationSplit="FROZEN_HOLDOUT"),
         "SCORE_REFERENCE_SPLIT_MISMATCH"),
        (lambda x: x["caseResults"].pop(),
         "SCORE_CASE_POPULATION_MISMATCH"),
        (lambda x: x["caseResults"].append(copy.deepcopy(x["caseResults"][0])),
         "SCORE_CASE_POPULATION_MISMATCH"),
    ):
        candidate = copy.deepcopy(baseline)
        mutation(candidate)
        with pytest.raises(ValueError, match=error):
            compare_scores(baseline, candidate)
