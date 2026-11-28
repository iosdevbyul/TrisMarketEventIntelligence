"""Integrated read-only diagnosis of saved market experiments and rule attempts."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tris_market_event_intelligence.autonomous_market_experience_cli import read_sealed
from tris_market_event_intelligence.prediction_failure_analysis_cli import _safe_dir, analyze
from tris_market_event_intelligence.reflection_quality_analysis_cli import audit


def classify_attempt(record: dict) -> str:
    if record.get("rule") is not None:
        return "CANDIDATE_ACCEPTED_FOR_RESEARCH"
    if record.get("candidateCount", 0):
        review = record.get("review") or {}
        if not review:
            return "CRITIC_RECORD_MISSING"
        return "CRITIC_DID_NOT_ACCEPT"
    errors = record.get("errors", [])
    if not errors:
        return "EMPTY_CANDIDATE_RESPONSE"
    if any(e.get("reason") == "INSUFFICIENT_DISTINCT_MIXED_EXPERIENCES" for e in errors):
        return "INSUFFICIENT_DISTINCT_MIXED_EXPERIENCES"
    # A successful empty answer after retries was not separately recorded in V2.
    return "VALIDATION_ERRORS_WITH_NO_FINAL_CANDIDATE"


def rule_audit(run_dir: Path) -> dict:
    report = read_sealed(run_dir / "report.json")
    history = report["ruleHistory"]
    details = []
    for i, record in enumerate(history):
        index = i * 10
        if read_sealed(run_dir / f"rule-{index}.json") != record:
            raise ValueError("Rule history checkpoint mismatch")
        review = record.get("review") or {}
        details.append({
            "batchStartIndex": index,
            "status": classify_attempt(record),
            "candidateCount": record.get("candidateCount", 0),
            "generationErrors": record.get("errors", []),
            "initialReview": review.get("initialReview"),
            "revisionReview": review.get("revisionReview"),
            "initialReviewErrors": review.get("initialReviewErrors", []),
            "revisionErrors": review.get("revisionErrors", []),
            "revisedRule": review.get("revisedRule"),
            "acceptedRule": record.get("rule"),
        })
    return {
        "attempts": len(details),
        "statusCounts": dict(Counter(x["status"] for x in details)),
        "generationErrors": dict(Counter(
            e.get("reason", "UNKNOWN") for x in details for e in x["generationErrors"]
        )),
        "details": details,
        "limitation": (
            "V2 did not persist rejected raw generation responses or all original "
            "candidate text. Retry exhaustion and later empty output may be indistinguishable."
        ),
    }


def run(run_dir: Path, seed: Path, prices: Path, output: Path) -> dict:
    prediction, rows = analyze(run_dir, prices)
    reflection = audit(run_dir, seed, prices, detail=rows)
    rules = rule_audit(run_dir)
    _safe_dir(output, exclude=[run_dir, seed, prices.parent])
    result = {
        "status": "DIAGNOSTICS_COMPLETED",
        "historicalResearchOnly": True,
        "promotionAllowed": False,
        "prospectiveAccuracy": None,
        "modelCalls": 0,
        "prediction": prediction,
        "reflection": reflection,
        "rules": rules,
        "predictionDetails": rows,
    }
    with (output / "research-diagnostics.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    return {
        "status": result["status"],
        "articles": prediction["articles"],
        "evaluatedRows": len(rows),
        "newExperiences": reflection["newExperiences"],
        "ruleAttempts": rules["attempts"],
        "ruleStatusCounts": rules["statusCounts"],
        "modelCalls": 0,
        "report": str(output / "research-diagnostics.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "seed", "prices", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.run, args.seed, args.prices, args.output_dir),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
