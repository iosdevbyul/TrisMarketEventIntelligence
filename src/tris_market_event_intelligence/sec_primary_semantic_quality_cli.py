"""Source-revalidated heuristic quality triage for unreviewed SEC semantic reports.

This offline audit is NOT a human judgment, evidence of causality, or a Gold label.
Only safe flag codes, case indices and immutable hashes leave private source JSON.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tris_market_event_intelligence.historical_issuer_corroboration_cli import _write
from tris_market_event_intelligence.sec_primary_semantic_research_cli import (
    MAX_CASES,
    MAX_INPUT,
)
from tris_market_event_intelligence.sec_primary_semantic_review_cli import (
    MAX_REPORT_BYTES,
    build_review_packet,
)
from tris_market_event_intelligence.sec_source_lineage_audit_cli import _read_json

MAX_REVIEW_BYTES = 1_000_000
CAUSAL = re.compile(
    r"\\b(?:caused|triggered|drove|driven|guaranteed|proves?|"
    r"definitely|certainly)\\b",
    re.IGNORECASE,
)
PRICE = re.compile(
    r"\\b(?:stock|share|price|returns?)\\s+"
    r"(?:will|must|is going to|would certainly)\\s+"
    r"(?:rise|fall|increase|decrease|surge|drop|rally)\\b",
    re.IGNORECASE,
)
CLOCK = re.compile(
    r"\\b(?:investors? (?:knew|saw|received)|"
    r"publicly available (?:before|at)|"
    r"already known (?:before|at))\\b",
    re.IGNORECASE,
)


def audit_semantic_quality(
    primary: dict, primary_sha: str, source_dir: Path,
    reports: list[tuple[dict, str]], review: dict, review_sha: str,
) -> dict:
    """Trust neither a stale review packet nor its model-generated prose."""
    expected = build_review_packet(primary, primary_sha, source_dir, reports)
    if (review != expected or not re.fullmatch(r"[a-f0-9]{64}", review_sha)
            or review.get("status") != "HUMAN_REVIEW_PENDING"):
        raise ValueError("SEC_SEMANTIC_QUALITY_REVIEW_PACKET_MISMATCH")
    findings: list[dict] = []
    counts = {"observations": 0, "hypotheses": 0, "summaries": 0}
    for case in expected["cases"]:
        index = case["caseIndex"]
        def check(location: str, value: str) -> None:
            for code, expression in (
                ("POTENTIAL_UNSUPPORTED_CAUSAL_LANGUAGE", CAUSAL),
                ("POTENTIAL_STOCK_DIRECTION_ASSERTION", PRICE),
                ("POTENTIAL_UNVERIFIED_PUBLICATION_CLAIM", CLOCK),
            ):
                if expression.search(value):
                    findings.append({
                        "caseIndex": index, "location": location, "code": code,
                    })
        for number, observation in enumerate(case["observations"]):
            counts["observations"] += 1
            check(f"observations[{number}].interpretation", observation["interpretation"])
        for number, hypothesis in enumerate(case["hypotheses"]):
            counts["hypotheses"] += 1
            check(f"hypotheses[{number}].mechanism", hypothesis["mechanism"])
            check(f"hypotheses[{number}].limitation", hypothesis["limitation"])
        counts["summaries"] += 1
        check("summary", case["summary"])
    return {
        "schemaVersion": "sec_primary_semantic_quality_triage_v1",
        "status": "HUMAN_SEMANTIC_REVIEW_REQUIRED",
        "primaryReportSha256": primary_sha,
        "humanReviewPacketSha256": review_sha,
        "casesAudited": len(expected["cases"]),
        "totalPinnedCases": expected["totalPinnedCases"],
        "remainingCaseIndexes": [
            item["caseIndex"] for item in expected["pendingCaseIndexes"]
        ],
        "counts": counts,
        "heuristicFindingCount": len(findings),
        "heuristicFindings": findings,
        "heuristicsAreNotTruthLabels": True,
        "zeroHeuristicFindingsDoesNotProveSemanticAccuracy": True,
        "humanDecisionsRecorded": 0,
        "reviewApproved": False,
        "allSourceQuotesReverified": True,
        "historicalPublicTimesIndependentlyVerified": 0,
        "trainingReady": False,
        "predictionReady": False,
        "promotionAllowed": False,
        "productionDbModified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline SEC human-review quality triage")
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True, action="append")
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage = "VERIFY_LOCAL_SEC_REPORTS"
    try:
        if len(args.report) > MAX_CASES:
            raise ValueError("SEC_SEMANTIC_QUALITY_REPORTS_INVALID")
        primary, primary_sha = _read_json(args.primary, MAX_INPUT)
        reports = [_read_json(path, MAX_REPORT_BYTES) for path in args.report]
        review, review_sha = _read_json(args.review, MAX_REVIEW_BYTES)
        audit = audit_semantic_quality(
            primary, primary_sha, args.source_dir, reports, review, review_sha,
        )
        stage = "WRITE_PRIVATE_QUALITY_TRIAGE"
        _write(args.output, audit)
        print(json.dumps({
            "status": audit["status"],
            "casesAudited": audit["casesAudited"],
            "remaining": len(audit["remainingCaseIndexes"]),
            "heuristicFindingCount": audit["heuristicFindingCount"],
            "humanDecisionsRecorded": 0,
            "trainingReady": False,
            "outputCreated": True,
        }))
    except (OSError, ValueError, KeyError, TypeError, UnicodeError,
            json.JSONDecodeError, AttributeError) as exc:
        code = (
            exc.args[0] if type(exc) in {ValueError, TypeError}
            and exc.args and isinstance(exc.args[0], str)
            and re.fullmatch(r"SEC_[A-Z0-9_]{8,90}", exc.args[0])
            else "SEC_SEMANTIC_QUALITY_INVALID_OR_UNSAFE_INPUT"
        )
        parser.exit(2, f"sec-semantic-quality: FAILED_STAGE={stage} ERROR_CODE={code}\n")


if __name__ == "__main__":
    main()
