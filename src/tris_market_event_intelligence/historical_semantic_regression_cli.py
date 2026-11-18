"""Compare two immutable semantic-score artifacts without any model calls.

This is a development-run regression diagnostic, not independent validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_METRICS_HIGHER = (
    "sourceCoverage",
    "requiredEvidenceRecall",
    "decisionAccuracy",
    "evidenceStatusAccuracy",
    "completionSafety",
    "primaryEvidenceAccuracy",
    "directRecordAccuracy",
    "indirectProgressAccuracy",
)
_METRICS_LOWER = ("criticalErrors", "modelCompletionClaimRate")


def compare_scores(baseline: dict, candidate: dict) -> dict:
    if baseline.get("schemaVersion") != candidate.get("schemaVersion"):
        raise ValueError("SCORE_SCHEMA_MISMATCH")
    if baseline.get("schemaVersion") not in {
        "historical_semantic_score_v2",
    }:
        raise ValueError("SCORE_SCHEMA_UNSUPPORTED")
    for key in ("referenceStatus", "evaluationSplit"):
        if baseline.get(key) != candidate.get(key):
            raise ValueError("SCORE_REFERENCE_SPLIT_MISMATCH:" + key)

    before = {x["id"]: x for x in baseline["caseResults"]}
    after = {x["id"]: x for x in candidate["caseResults"]}
    if not before or set(before) != set(after) or len(before) != len(
        baseline["caseResults"]
    ) or len(after) != len(candidate["caseResults"]):
        raise ValueError("SCORE_CASE_POPULATION_MISMATCH")

    deltas = {}
    regressions = []
    improvements = []
    for name in (*_METRICS_HIGHER, *_METRICS_LOWER):
        old = baseline["rates"][name]
        new = candidate["rates"][name]
        delta = round(new - old, 10)
        deltas[name] = {"baseline": old, "candidate": new, "delta": delta}
        lower_is_better = name in _METRICS_LOWER
        if (delta > 0 and lower_is_better) or (
            delta < 0 and not lower_is_better
        ):
            regressions.append(name)
        elif delta:
            improvements.append(name)

    newly_failing = sorted(
        name for name in before
        if before[name]["status"] == "PASS"
        and after[name]["status"] != "PASS"
    )
    newly_passing = sorted(
        name for name in before
        if before[name]["status"] != "PASS"
        and after[name]["status"] == "PASS"
    )
    changed_errors = [
        {
            "id": name,
            "removed": sorted(
                set(before[name].get("errors", []))
                - set(after[name].get("errors", []))
            ),
            "added": sorted(
                set(after[name].get("errors", []))
                - set(before[name].get("errors", []))
            ),
        }
        for name in sorted(before)
        if set(before[name].get("errors", [])) != set(
            after[name].get("errors", [])
        )
    ]
    return {
        "schemaVersion": "historical_semantic_regression_v1",
        "status": (
            "REGRESSION_DETECTED" if regressions or newly_failing
            else "NO_MEASURED_REGRESSION"
        ),
        "baselineEvaluationSplit": baseline["evaluationSplit"],
        "modelCalls": 0,
        "metrics": deltas,
        "regressedMetrics": regressions,
        "improvedMetrics": improvements,
        "newlyFailingCases": newly_failing,
        "newlyPassingCases": newly_passing,
        "changedErrors": changed_errors,
        "trustVerified": False,
        "limitations": [
            "Comparisons are restricted to equal schema, reference status, "
            "evaluation split and case identities.",
            "Development references were repeatedly used for model tuning.",
            "A stable or improved result does not establish independent "
            "reasoning quality, historical PIT or generalization.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    report = compare_scores(baseline, candidate)
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        if args.output.exists() or args.output.is_symlink():
            parser.error("OUTPUT_ALREADY_EXISTS:" + str(args.output))
        with args.output.open("x", encoding="utf-8") as file:
            file.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
