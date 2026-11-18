"""Finalize explicit human evaluation of a saved private market-pressure review.

No LLM calls or DB writes. Human validation is per hypothesis and does not
automatically turn a model result into an observed market-return forecast.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

_CHECKS = (
    "sourceFactSupported",
    "targetExposurePlausible",
    "transmissionMechanismPlausible",
    "directionSupported",
    "strengthSupported",
    "countervailingFactorsConsidered",
)
_DECISIONS = {"SUPPORTED_WITH_LIMITATIONS", "NOT_SUPPORTED", "INCONCLUSIVE"}


def finalize_review(review: dict, *, reviewer: str) -> dict:
    if (
        review.get("schemaVersion") != "market_pressure_semantic_review_v1"
        or review.get("economicAccuracyVerified") is not False
        or not isinstance(review.get("reviewItems"), list)
        or not review["reviewItems"]
    ):
        raise ValueError("INVALID_SEMANTIC_REVIEW_INPUT")
    if not 2 <= len(reviewer.strip()) <= 80:
        raise ValueError("REVIEWER_NAME_REQUIRED")
    items = []
    summary = {name: 0 for name in sorted(_DECISIONS)}
    for item in review["reviewItems"]:
        decision = item["humanReview"]
        if (
            decision.get("humanDecision") not in _DECISIONS
            or any(type(decision.get(key)) is not bool for key in _CHECKS)
            or not isinstance(decision.get("reason"), str)
            or len(decision["reason"].strip()) < 30
        ):
            raise ValueError("ALL_HUMAN_EVALUATIONS_AND_REASON_REQUIRED")
        if (
            decision["humanDecision"] == "SUPPORTED_WITH_LIMITATIONS"
            and not all(decision[field] for field in _CHECKS)
        ):
            raise ValueError("HUMAN_SUPPORTED_BUT_CRITERIA_FAILED")
        summary[decision["humanDecision"]] += 1
        items.append({
            "eventId": item["eventId"],
            "proposalIndex": item["proposalIndex"],
            "validatorDisposition": item["validatorDisposition"],
            "humanReview": decision,
        })
    canonical = json.dumps(
        review, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    )
    return {
        "schemaVersion": "market_pressure_human_evaluation_v1",
        "status": "HUMAN_REVIEW_COMPLETE",
        "reviewer": reviewer.strip(),
        "sourceTraceSha256": review["traceSha256"],
        "reviewFileSha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "semanticAccuracyVerifiedByIndependentStudy": False,
        "modelCalls": 0,
        "dbWrites": 0,
        "judgmentCounts": summary,
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reviewed = json.loads(args.review.read_text(encoding="utf-8"))
    result = finalize_review(reviewed, reviewer=args.reviewer)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, ensure_ascii=False)
        out.write("\n")
    print(json.dumps({
        "status": result["status"],
        "reviewed": len(result["items"]),
        "outputCreated": True, "modelCalls": 0, "dbWrites": 0,
    }))


if __name__ == "__main__":
    main()
