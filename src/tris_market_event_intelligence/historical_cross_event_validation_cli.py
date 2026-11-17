"""Source-locked, no-model cross-event evaluation of independent histories.

Does not equate good reference coverage with economic or semantic truth.
Differences in response schema are treated as noncomparable, never hidden.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tris_market_event_intelligence.historical_cross_domain_official_study_cli import (
    _require_new_output,
)
from tris_market_event_intelligence.historical_event_chain_explanation_quality import (
    audit_sequence,
)
from tris_market_event_intelligence.historical_independent_news_cli import (
    _digest,
    _read,
    _write,
)
from tris_market_event_intelligence.historical_official_event_chain_cli import (
    validate_chain,
)


def validate_independence(primary: dict, holdout: dict) -> None:
    validate_chain(primary)
    validate_chain(holdout)
    if primary["eventId"] == holdout["eventId"]:
        raise ValueError("CROSS_EVENT_SAME_EVENT_ID")
    first = {row["sourceId"] for row in primary["sources"]}
    second = {row["sourceId"] for row in holdout["sources"]}
    first_urls = {row["url"] for row in primary["sources"]}
    second_urls = {row["url"] for row in holdout["sources"]}
    first_domains = {row["publisher"] for row in primary["sources"]}
    second_domains = {row["publisher"] for row in holdout["sources"]}
    if (
        first & second or first_urls & second_urls
        or first_domains & second_domains
        or first_domains != {"FAA"}
        or second_domains != {"FDA"}
    ):
        raise ValueError("CROSS_EVENT_SOURCE_INDEPENDENCE_FAILED")


def summarize_sequence(original: dict, report: dict) -> dict:
    items = []
    for step, diagnosis in zip(
        original["replaySteps"], report["replayStepAudits"], strict=True,
    ):
        quality = diagnosis["qualityAudit"]
        assessment = step["assessment"]
        count = assessment.get("availableSourceClaimCount")
        matched = assessment.get("sourceClaimCoverage")
        edge_total = assessment.get("visibleEdgeCount")
        edge_matched = assessment.get("visibleEdgeCoverage")
        items.append({
            "asOfDateExclusive": step["asOfDateExclusive"],
            "sourceClaimReferences": matched,
            "eligibleSourceClaims": count,
            "curatedEdgeReferences": edge_matched,
            "visibleCuratedEdges": edge_total,
            "allReferencesCovered": (
                matched == count and edge_matched == edge_total
            ),
            "structurallyRejectedFacts": len(
                assessment.get("rejectedFactInterpretations", [])
            ),
            "structurallyRejectedEdges": len(
                assessment.get("rejectedEdgeExplanations", [])
            ),
            "explanationIssueCodes": quality["issueCodes"],
            "unansweredPendingDecisionClaims": quality[
                "unansweredPendingDecisionClaims"
            ],
            "missingRecentSourceIds": quality["missingRecentSourceIds"],
            "circularMissingEvidenceCount": sum(
                "MISSING_EVIDENCE_RESTATES_KNOWN_EDGE_BASIS" in row["issueCodes"]
                for row in quality["edgeEvidenceReviews"]
            ),
            "allExplanationsSemanticallyVerified": False,
        })
    return {
        "eventId": original["eventId"],
        "sourceCohortSha256": original["sourceCohortSha256"],
        "researchResultSchema": original["schemaVersion"],
        "steps": items,
        "originalResearchModelCalls": original.get("modelCalls"),
        "auditModelCalls": 0,
        "semanticAccuracyVerified": False,
        "historicalPITVerified": False,
    }


def compare_independent_events(
    primary: dict, primary_saved: dict, holdout: dict, holdout_saved: dict,
) -> dict:
    validate_independence(primary, holdout)
    primary_audit = audit_sequence(primary, primary_saved)
    holdout_audit = audit_sequence(holdout, holdout_saved)
    a = summarize_sequence(primary_saved, primary_audit)
    b = summarize_sequence(holdout_saved, holdout_audit)
    same_schema = a["researchResultSchema"] == b["researchResultSchema"]
    return {
        "schemaVersion": "historical_cross_event_qualitative_validation_v1",
        "status": "INDEPENDENT_EVENTS_REQUIRE_HUMAN_SEMANTIC_REVIEW",
        "primary": a,
        "independentHoldout": b,
        "sourceIndependence": {
            "distinctPublishers": True,
            "sourceIdOverlap": 0,
            "sourceUrlOverlap": 0,
            "primaryInputSha256": _digest(primary),
            "holdoutInputSha256": _digest(holdout),
        },
        "fairLikeForLikeModelSchemaComparison": same_schema,
        "evidenceOfCrossEventGeneralization": False,
        "referenceCoverageIsNotSemanticAccuracy": True,
        "historicalPITVerified": False,
        "economicCausalityVerified": False,
        "semanticAccuracyVerified": False,
        "auditModelCalls": 0,
        "promotionAllowed": False,
        "tradesPlaced": False,
        "originalArtifactsUnchanged": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FAA vs independent FDA event chains")
    parser.add_argument("--primary-chain", type=Path, required=True)
    parser.add_argument("--primary-existing", type=Path, required=True)
    parser.add_argument("--holdout-chain", type=Path, required=True)
    parser.add_argument("--holdout-existing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require_new_output(args.output)
    result = compare_independent_events(
        _read(args.primary_chain), _read(args.primary_existing),
        _read(args.holdout_chain), _read(args.holdout_existing),
    )
    _write(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "fairLikeForLikeModelSchemaComparison": result[
            "fairLikeForLikeModelSchemaComparison"
        ],
        "primaryEventId": result["primary"]["eventId"],
        "holdoutEventId": result["independentHoldout"]["eventId"],
        "auditModelCalls": 0,
        "historicalPITVerified": False,
        "evidenceOfCrossEventGeneralization": False,
        "outputCreated": True,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
