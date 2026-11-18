"""Unit tests for model-independent retrospective semantic benchmark scoring."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from tris_market_event_intelligence.historical_semantic_quality_cli import (
    score_runs,
)


def benchmark() -> dict:
    return json.loads(Path(
        "research/historical-semantic-benchmark-v1.json"
    ).read_text(encoding="utf-8"))


def synthetic_runs(reference: dict) -> dict:
    runs: dict = {}
    for event in ("FAA", "FDA"):
        event_cases = [case for case in reference["cases"]
                       if case["event"] == event]
        steps = []
        for date in sorted({case["date"] for case in event_cases}):
            cases = [case for case in event_cases if case["date"] == date]
            screens = []
            axes = []
            all_claims = sorted({
                tuple(ref) for case in cases for ref in case["required"]
            })
            for case in cases:
                selected = [{"sourceId": ref[0], "claimId": ref[1]}
                            for ref in case["required"]]
                screens.append({
                    "questionId": case["questionId"],
                    "allVisibleClaimRoles": [
                        {"sourceId": ref[0], "claimId": ref[1],
                         "role": (
                             "INTERMEDIATE_DECISION"
                             if ref in map(tuple, case["required"])
                             else "NOT_RELEVANT"
                         )}
                        for ref in all_claims
                    ],
                })
                axes.append({
                    "questionId": case["questionId"],
                    "supportingEvidence": selected,
                    "regulatoryDecisionStage": case["decision"][0],
                    "requestedEvidenceStatus": case["evidence"][0],
                    "implementationCompletionStatus": case["completion"][0],
                })
            steps.append({
                "asOfDateExclusive": date,
                "assessment": {
                    "sourceScopedComposition": {"sourceCoverage": {
                        "requiredClaimRefs": [
                            {"sourceId": ref[0], "claimId": ref[1]}
                            for ref in all_claims
                        ],
                        "explainedClaimRefs": [
                            {"sourceId": ref[0], "claimId": ref[1]}
                            for ref in all_claims
                        ],
                    }},
                    "independentQuestionAxes": {
                        "questionAssessments": axes,
                        "questionEvidenceScreens": screens,
                    },
                },
            })
        runs[event] = {"historicalPITVerified": False, "replaySteps": steps}
    return runs


def test_all_reference_expectations_pass_but_trust_still_blocked() -> None:
    reference = benchmark()
    report = score_runs(reference, synthetic_runs(reference))
    assert report["rates"]["requiredEvidenceRecall"] == 1
    assert report["rates"]["decisionAccuracy"] == 1
    assert report["rates"]["evidenceStatusAccuracy"] == 1
    assert report["rates"]["completionSafety"] == 1
    assert report["rates"]["sourceCoverage"] == 1
    assert report["rates"]["criticalErrors"] == 0
    assert report["status"] == "NOT_YET_TRUSTWORTHY"
    assert not report["trustGate"]["eventFamiliesSufficient"]
    assert not report["trustGate"]["sampleSizeSufficient"]
    assert not report["trustGate"]["historicalPITVerified"]
    assert not report["trustGate"]["independentGoldReviewComplete"]
    assert not report["trustGate"]["evaluationSplitIsFrozenHoldout"]


def test_scoring_catches_missing_later_decisions_and_false_completion() -> None:
    reference = benchmark()
    runs = synthetic_runs(reference)
    late = next(
        step for step in runs["FAA"]["replaySteps"]
        if step["asOfDateExclusive"] == "2024-01-25"
    )
    item = next(x for x in late["assessment"]["independentQuestionAxes"][
        "questionAssessments"
    ] if x["questionId"] == "faa-2024-boeing-production-oversight-outcome")
    item["supportingEvidence"] = []
    item["regulatoryDecisionStage"] = "ANNOUNCED"
    item["implementationCompletionStatus"] = "COMPLETED"
    report = score_runs(reference, runs)
    case = next(x for x in report["caseResults"]
                if x["id"] == "faa-late-oversight")
    assert "REQUIRED_CLAIM_MISSING" in case["errors"]
    assert "REGULATORY_DECISION_MISMATCH" in case["errors"]
    assert "COMPLETION_SAFETY_FAILURE" in case["errors"]
    assert report["rates"]["criticalErrors"] >= 3


def test_scoring_fails_on_future_leakage_and_missing_case() -> None:
    reference = benchmark()
    runs = synthetic_runs(reference)
    early = runs["FDA"]["replaySteps"][0]
    item = early["assessment"]["independentQuestionAxes"][
        "questionAssessments"
    ][0]
    item["supportingEvidence"].append({
        "sourceId": "fda-20230911-mrna",
        "claimId": "mrna-authorization",
    })
    report = score_runs(reference, runs)
    case = next(x for x in report["caseResults"]
                if x["id"] == "fda-early-formula")
    assert "FUTURE_CLAIM_LEAKAGE" in case["errors"]
    assert "INVISIBLE_CLAIM_REFERENCE" in case["errors"]
    missing_runs = copy.deepcopy(runs)
    missing_runs["FAA"]["replaySteps"][0]["assessment"][
        "independentQuestionAxes"
    ]["questionAssessments"].pop()
    missing_report = score_runs(reference, missing_runs)
    assert missing_report["counts"]["missingCases"] == 1
    assert missing_report["status"] == "NOT_YET_TRUSTWORTHY"
