"""Private, source-linked semantic audit of actual model market-pressure hypotheses.

This does NOT use a second LLM as a judge. Verbatim evidence, model reasoning,
opposing factors and limits are placed side by side for human review.
Accepted Pydantic output is NOT labeled economically correct.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tris_market_event_intelligence.market_pressure_replay_cli import replay_trace


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _related_model_calls(rows: list[dict]) -> dict[int, dict]:
    calls: dict[int, dict] = {}
    for row in rows:
        number = row.get("call")
        if not isinstance(number, int):
            continue
        slot = calls.setdefault(number, {})
        if row.get("event") == "LLM_REQUEST":
            slot["request"] = row
        elif row.get("event") == "LLM_RESPONSE":
            slot["response"] = row
    return calls


def semantic_review(trace: Path) -> dict:
    """Read saved trace; run only deterministic local validators, no new model."""
    raw = trace.read_text(encoding="utf-8")
    replay = replay_trace(trace)
    calls = _related_model_calls(_read_jsonl(trace))
    reviews: list[dict] = []
    for call_id, stage in sorted(calls.items()):
        request, response = stage.get("request"), stage.get("response")
        if not request or not response or request.get("stage") != "IMPACT_GENERATION":
            continue
        payload = json.loads(request["userInput"])
        grounded = payload.get("groundedClaims", [])
        generated = json.loads(response["rawResponse"])
        proposals = generated["impacts"]
        event_id = payload["eventId"]
        match = next(
            (record for record in replay["events"] if record["eventId"] == event_id),
            None,
        )
        if match is None:
            raise ValueError("REVIEW_EVENT_NOT_IN_REPLAY")
        accepted = {
            item["proposalIndex"] for item in match["acceptedImpacts"]
        }
        rejected = {
            item["index"]: item for item in match["rejectedImpacts"]
        }
        for index, item in enumerate(proposals):
            if not isinstance(item, dict):
                continue
            references: list[dict] = []
            for source_index in item.get("claimIndexes", []):
                if type(source_index) is not int or not 0 <= source_index < len(grounded):
                    continue
                claim = grounded[source_index]
                references.append({
                    "claimIndex": source_index,
                    "sourceId": claim["sourceId"],
                    "publishedSourceVerification": claim["sourceVerification"],
                    "factKind": claim["factKind"],
                    "novelty": claim["novelty"],
                    "quotedObservation": claim["excerpt"],
                    "indicatorValue": claim.get("indicatorValue"),
                    "indicatorPrevious": claim.get("indicatorPrevious"),
                    "indicatorExpectation": claim.get("indicatorExpectation"),
                })
            missing_expectation = (
                item.get("expectationComparison") in {
                    "ABOVE_EXPECTATION", "BELOW_EXPECTATION", "IN_LINE",
                }
                and not any(
                    ref.get("indicatorExpectation") is not None
                    for ref in references
                )
            )
            flags: list[str] = []
            if not item.get("countervailingFactors"):
                flags.append("COUNTERVAILING_FACTORS_NOT_SPECIFIED")
            if item.get("strength") in {"MODERATE", "HIGH"}:
                flags.append("STRENGTH_REQUIRES_INDEPENDENT_SEMANTIC_REVIEW")
            if len({ref["sourceId"] for ref in references}) == 1:
                flags.append("SINGLE_CITED_SOURCE_NOT_INDEPENDENT_CORROBORATION")
            if all(
                ref["publishedSourceVerification"] == "REPORTED_UNVERIFIED"
                for ref in references
            ):
                flags.append("REPORTED_CLAIMS_NOT_INDEPENDENTLY_VERIFIED")
            if missing_expectation:
                flags.append("CONSENSUS_COMPARISON_NOT_SUPPORTED")
            if item.get("direction") == "MIXED":
                flags.append("MIXED_DIRECTION_NEEDS_OPPOSING_CHANNELS_EXPLAINED")
            if item.get("direction") == "UNCERTAIN":
                flags.append("UNKNOWN_DIRECTION_NOT_A_NEGATIVE_SIGNAL")
            reviews.append({
                "eventId": event_id,
                "proposalIndex": index,
                "validatorDisposition": (
                    "ACCEPTED_REQUIRES_SEMANTIC_REVIEW" if index in accepted
                    else "REJECTED_BY_GUARD"
                ),
                "rejectionCode": rejected.get(index, {}).get("code"),
                "targetType": item.get("targetType"),
                "targetName": item.get("targetName"),
                "market": item.get("market"),
                "sector": item.get("sector"),
                "direction": item.get("direction"),
                "strength": item.get("strength"),
                "strengthRationale": item.get("strengthRationale"),
                "horizon": item.get("horizon"),
                "transmissionChannel": item.get("transmissionChannel"),
                "mechanismRationale": item.get("rationale"),
                "countervailingFactors": item.get("countervailingFactors", []),
                "modelDrivers": item.get("drivers", []),
                "quotedEvidence": item.get("evidence_excerpt"),
                "references": references,
                "reviewFlags": flags,
                "humanReview": {
                    "sourceFactSupported": None,
                    "targetExposurePlausible": None,
                    "transmissionMechanismPlausible": None,
                    "directionSupported": None,
                    "strengthSupported": None,
                    "countervailingFactorsConsidered": None,
                    "humanDecision": "PENDING",
                    "reason": "",
                },
            })
    return {
        "schemaVersion": "market_pressure_semantic_review_v1",
        "status": "HUMAN_REVIEW_REQUIRED",
        "traceSha256": _hash_text(raw),
        "readOnly": True,
        "newModelCalls": 0,
        "dbWrites": 0,
        "acceptedBySchemaCount": replay["acceptedImpactCount"],
        "rejectedBySchemaCount": replay["rejectedImpactCount"],
        "economicAccuracyVerified": False,
        "humanReviewedCount": 0,
        "reviewItems": reviews,
    }


def _create_only(path: Path, report: dict) -> None:
    with path.open("x", encoding="utf-8") as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
        out.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _create_only(args.output, semantic_review(args.trace))
    print(json.dumps({
        "status": "HUMAN_REVIEW_REQUIRED",
        "acceptedBySchemaCount": semantic_review(args.trace)["acceptedBySchemaCount"],
        "outputCreated": True,
        "modelCalls": 0,
        "dbWrites": 0,
        "output": str(args.output),
    }))


if __name__ == "__main__":
    main()
