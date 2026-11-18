"""Score saved historical replay artifacts against separate reviewer references.

This CLI NEVER calls an LLM and is not imported by the replay pipeline.
FAA/FDA development annotations are not a frozen, independently scored holdout.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _pairs(items: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(x["sourceId"], x["claimId"]) for x in items}


def score_runs(benchmark: dict, runs: dict[str, dict]) -> dict:
    if benchmark.get("schemaVersion") not in {
        "historical_semantic_benchmark_v1",
        "historical_semantic_benchmark_v2",
    }:
        raise ValueError("BENCHMARK_SCHEMA_UNSUPPORTED")
    index: dict[tuple[str, str, str], tuple[dict, dict]] = {}
    event_pit: dict[str, bool] = {}
    source_coverage_checks: list[bool] = []
    for event, run in runs.items():
        event_pit[event] = run.get("historicalPITVerified") is True
        for step in run["replaySteps"]:
            date = step["asOfDateExclusive"]
            assessment = step["assessment"]
            axes = assessment["independentQuestionAxes"]
            source_coverage = assessment["sourceScopedComposition"]["sourceCoverage"]
            source_coverage_checks.append(
                _pairs(source_coverage["requiredClaimRefs"])
                == _pairs(source_coverage["explainedClaimRefs"])
                and len(source_coverage["requiredClaimRefs"])
                == len(source_coverage["explainedClaimRefs"])
            )
            screens = {
                x["questionId"]: x
                for x in axes["questionEvidenceScreens"]
            }
            for result in axes["questionAssessments"]:
                key = (event, date, result["questionId"])
                if key in index:
                    raise ValueError("DUPLICATE_ASSESSMENT:" + repr(key))
                index[key] = (result, screens[result["questionId"]])
    is_v2 = benchmark["schemaVersion"] == "historical_semantic_benchmark_v2"
    details: list[dict] = []
    totals = defaultdict(int)
    for case in benchmark["cases"]:
        key = (case["event"], case["date"], case["questionId"])
        if key not in index:
            details.append({"id": case["id"], "status": "MISSING_ASSESSMENT"})
            totals["missingCases"] += 1
            continue
        result, screen = index[key]
        supporting = _pairs(result["supportingEvidence"])
        visible = _pairs(screen["allVisibleClaimRoles"])
        required = set(map(tuple, case["required"]))
        forbidden = set(map(tuple, case["forbidden"]))
        included = supporting & required
        missing = required - supporting
        leaking = supporting & forbidden
        invalid = supporting - visible
        role_by_ref = {
            (x["sourceId"], x["claimId"]): x["role"]
            for x in screen["allVisibleClaimRoles"]
        }
        incorrectly_rejected = sorted([
            list(x) for x in required
            if role_by_ref.get(x) == "NOT_RELEVANT"
        ])
        decision_ok = result.get("regulatoryDecisionStage") in case["decision"]
        evidence_ok = result.get("requestedEvidenceStatus") in case["evidence"]
        completion_ok = result.get("implementationCompletionStatus") in case["completion"]
        direct_ok = (
            result.get("directRecordAvailability")
            in case.get("directRecordAvailability", [])
        ) if is_v2 else None
        indirect_ok = (
            result.get("indirectRegulatoryProgress")
            in case.get("indirectRegulatoryProgress", [])
        ) if is_v2 else None
        model_completion_reported = (
            result.get("modelReportedImplementationCompletionStatus")
            == "COMPLETED"
        ) if is_v2 else None
        # Hard safety failure: any COMPLETED decision is not sufficiently
        # grounded until explicit independent source-level completion evidence
        # verification exists. The current mode deliberately never claims it.
        completion_safe = (
            completion_ok and
            result.get("implementationCompletionStatus") != "COMPLETED"
        )
        for name, value in (
            ("requiredEvidence", len(required)),
            ("recalledEvidence", len(included)),
            ("decisionCorrect", int(decision_ok)),
            ("evidenceStatusCorrect", int(evidence_ok)),
            ("completionSafe", int(completion_safe)),
            ("evaluatedCases", 1),
        ):
            totals[name] += value
        if is_v2:
            totals["directRecordCorrect"] += int(bool(direct_ok))
            totals["indirectProgressCorrect"] += int(bool(indirect_ok))
            totals["modelCompletionClaims"] += int(bool(model_completion_reported))
        errors = []
        if missing:
            errors.append("REQUIRED_CLAIM_MISSING")
        if leaking:
            errors.append("FUTURE_CLAIM_LEAKAGE")
        if invalid:
            errors.append("INVISIBLE_CLAIM_REFERENCE")
        if incorrectly_rejected:
            errors.append("RELEVANT_CLAIM_SCREENED_OUT")
        if not decision_ok:
            errors.append("REGULATORY_DECISION_MISMATCH")
        if not evidence_ok:
            errors.append("REQUESTED_EVIDENCE_STATUS_MISMATCH")
        if not completion_safe:
            errors.append("COMPLETION_SAFETY_FAILURE")
        if is_v2:
            if not direct_ok:
                errors.append("DIRECT_RECORD_STATUS_MISMATCH")
            if not indirect_ok:
                errors.append("INDIRECT_PROGRESS_STATUS_MISMATCH")
            if model_completion_reported:
                errors.append("MODEL_UNVERIFIED_COMPLETION_CLAIM")
        totals["criticalErrors"] += len(errors)
        details.append({
            "id": case["id"],
            "status": "PASS" if not errors else "REVIEW_REQUIRED",
            "requiredClaimRecall": (
                len(included) / len(required) if required else 1.0
            ),
            "missingRequired": sorted(map(list, missing)),
            "forbiddenReferences": sorted(map(list, leaking)),
            "invisibleReferences": sorted(map(list, invalid)),
            "incorrectlyRejected": incorrectly_rejected,
            "decisionObserved": result.get("regulatoryDecisionStage"),
            "decisionAllowed": case["decision"],
            "evidenceObserved": result.get("requestedEvidenceStatus"),
            "evidenceAllowed": case["evidence"],
            "completionObserved": result.get("implementationCompletionStatus"),
            "completionAllowed": case["completion"],
            "warnings": result.get("evidenceStateConsistencyFlags", []),
            **({
                "directRecordObserved": result.get("directRecordAvailability"),
                "directRecordAllowed": case["directRecordAvailability"],
                "indirectProgressObserved": result.get("indirectRegulatoryProgress"),
                "indirectProgressAllowed": case["indirectRegulatoryProgress"],
                "modelReportedCompletion": result.get(
                    "modelReportedImplementationCompletionStatus"
                ),
            } if is_v2 else {}),
            "errors": errors,
        })
    denominator = totals["evaluatedCases"] or 1
    required = totals["requiredEvidence"] or 1
    rates = {
        "sourceCoverage": (
            sum(source_coverage_checks) / len(source_coverage_checks)
            if source_coverage_checks else 0.0
        ),
        "requiredEvidenceRecall": totals["recalledEvidence"] / required,
        "decisionAccuracy": totals["decisionCorrect"] / denominator,
        "evidenceStatusAccuracy": (
            totals["evidenceStatusCorrect"] / denominator
        ),
        "completionSafety": totals["completionSafe"] / denominator,
        "criticalErrors": totals["criticalErrors"],
    }
    if is_v2:
        rates.update({
            "directRecordAccuracy": totals["directRecordCorrect"] / denominator,
            "indirectProgressAccuracy": (
                totals["indirectProgressCorrect"] / denominator
            ),
            "modelCompletionClaimRate": (
                totals["modelCompletionClaims"] / denominator
            ),
        })
    thresholds = benchmark["thresholds"]
    threshold_checks = {
        key: (
            value <= thresholds[key]
            if key in {"criticalErrors", "modelCompletionClaimRate"}
            else value >= thresholds[key]
        )
        for key, value in rates.items()
    }
    enough_families = len({
        case["event"] for case in benchmark["cases"]
    }) >= benchmark["minimumIndependentEventFamiliesForTrust"]
    enough_cases = (
        totals["evaluatedCases"]
        >= benchmark["minimumQuestionAssessmentsForTrust"]
        and totals["missingCases"] == 0
    )
    pit_verified = all(event_pit.values())
    trust_gate = {
        "benchmarkThresholdsPassed": all(threshold_checks.values()),
        "eventFamiliesSufficient": enough_families,
        "sampleSizeSufficient": enough_cases,
        "historicalPITVerified": pit_verified,
        "independentGoldReviewComplete": (
            benchmark.get("reviewStatus") == "INDEPENDENTLY_VERIFIED"
        ),
        "evaluationSplitIsFrozenHoldout": (
            benchmark.get("evaluationSplit") == "FROZEN_HOLDOUT"
        ),
    }
    return {
        "schemaVersion": (
            "historical_semantic_score_v2" if is_v2
            else "historical_semantic_score_v1"
        ),
        "referenceStatus": benchmark["reviewStatus"],
        "evaluationSplit": benchmark.get("evaluationSplit", "UNSPECIFIED"),
        "status": (
            "RESEARCH_TRUST_CANDIDATE"
            if all(trust_gate.values()) else "NOT_YET_TRUSTWORTHY"
        ),
        "rates": rates,
        "counts": dict(totals),
        "thresholdChecks": threshold_checks,
        "trustGate": trust_gate,
        "caseResults": details,
        "limitations": [
            "Retrospective curator paraphrases are not archived historical PIT.",
            "Matching an answer label is not evidence of independent causal truth.",
            "FAA/FDA cases were used during development; only unseen frozen holdouts support generalization.",
            "This scorer is isolated from model inputs and never revises them.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--faa", type=Path, required=True)
    parser.add_argument("--fda", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    runs = {
        "FAA": json.loads(args.faa.read_text(encoding="utf-8")),
        "FDA": json.loads(args.fda.read_text(encoding="utf-8")),
    }
    report = score_runs(benchmark, runs)
    output = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        if args.output.exists() or args.output.is_symlink():
            parser.error("OUTPUT_ALREADY_EXISTS:" + str(args.output))
        with args.output.open("x", encoding="utf-8") as file:
            file.write(output)
    print(output, end="")


if __name__ == "__main__":
    main()
