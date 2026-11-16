"""Offline SEC claim-to-source review preparation; no automated truth labels.

This tool rechecks the pinned original and validated interpretations before
materializing PRIVATE, create-only, unreviewed claim rows for human adjudication.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tris_market_event_intelligence.historical_issuer_corroboration_cli import _write
from tris_market_event_intelligence.sec_primary_semantic_research_cli import MAX_CASES, MAX_INPUT
from tris_market_event_intelligence.sec_primary_semantic_review_cli import (
    MAX_REPORT_BYTES,
    build_review_packet,
)
from tris_market_event_intelligence.sec_source_lineage_audit_cli import _read_json

CLAIM_FIELDS = ("mechanism", "limitation")
MAX_OUTPUT_CLAIMS = MAX_CASES * (3 + 6 + 1)


def build_claim_review(
    primary: dict, primary_sha: str, source_dir: Path,
    reports: list[tuple[dict, str]],
) -> dict:
    """Revalidate original SEC bytes and each LLM quote before claim extraction."""
    verified = build_review_packet(primary, primary_sha, source_dir, reports)
    claims: list[dict] = []
    abstained_cases: list[int] = []
    for case in verified["cases"]:
        if case.get("analysisStatus") in {
            "INSUFFICIENT_SUBSTANTIVE_EVIDENCE",
            "REFERENT_UNRESOLVED",
            "DESCRIPTIVE_REVENUE_SHARE_ONLY",
        }:
            abstained_cases.append(case["caseIndex"])
            continue
        index = case["caseIndex"]
        anchors = [
            {"evidenceId": ob["evidence_id"], "exactSourceQuote": ob["verbatim_quote"]}
            for ob in case["observations"]
        ]
        fields: list[tuple[str, str, str]] = []
        for i, observation in enumerate(case["observations"]):
            fields.append((
                f"observations[{i}].interpretation",
                observation["interpretation"],
                "ISSUER_OBSERVATION",
            ))
        for i, hypothesis in enumerate(case["hypotheses"]):
            for field in CLAIM_FIELDS:
                fields.append((
                    f"hypotheses[{i}].{field}",
                    hypothesis[field],
                    hypothesis["category"],
                ))
        fields.append(("summary", case["summary"], "SUMMARY"))
        for location, statement, kind in fields:
            # Candidate evidence is exact-source and reviewer-visible only;
            # this does NOT assert semantic entailment.
            if "sourcePassages" in case:
                if location.startswith("observations["):
                    position = int(location.split("[", 1)[1].split("]", 1)[0])
                    bound_ids = [case["observations"][position]["evidence_id"]]
                elif location.startswith("hypotheses["):
                    position = int(location.split("[", 1)[1].split("]", 1)[0])
                    bound_ids = case["hypotheses"][position]["evidence_ids"]
                else:
                    bound_ids = [a["evidenceId"] for a in anchors]
                candidates = [a for a in anchors if a["evidenceId"] in bound_ids]
            else:
                candidates = (
                    anchors if kind in {
                        "ISSUER_OBSERVATION", "ISSUER_FUNDAMENTALS", "SUMMARY",
                    } else []
                )
            claims.append({
                "caseIndex": index,
                "accession": case["accession"],
                "sourceSha256": case["sourceSha256"],
                "semanticReportSha256": case["semanticReportSha256"],
                "location": location,
                "claimCategory": kind,
                "modelStatement": statement,
                "sourceExcerpt": case["sourceExcerpt"],
                **({"sourceExcerpts": case["sourceExcerpts"],
                    "sourcePassages": case["sourcePassages"]}
                   if "sourcePassages" in case else {}),
                "candidateEvidence": candidates,
                "independentIndustryEvidenceAvailable": False,
                "evidenceStatus": "NOT_ASSESSED",
                "humanEvidenceSupported": None,
                "humanSpeculationClearlyLabeled": None,
                "humanUnsupportedConcreteFacts": None,
                "humanNotes": "",
            })
    if len(claims) > MAX_OUTPUT_CLAIMS:
        raise ValueError("SEC_CLAIM_REVIEW_CARDINALITY_INVALID")
    return {
        "schemaVersion": "sec_primary_claim_evidence_review_v1",
        "status": "HUMAN_CLAIM_EVIDENCE_REVIEW_PENDING",
        "primaryReportSha256": primary_sha,
        "casesIncluded": verified["reviewCasesPrepared"],
        "totalPinnedCases": verified["totalPinnedCases"],
        "remainingCaseIndexes": [
            item["caseIndex"] for item in verified["pendingCaseIndexes"]
        ],
        "claimCount": len(claims),
        "abstainedCaseIndexes": abstained_cases,
        "claims": claims,
        "humanDecisionsRecorded": 0,
        "automatedSemanticEntailmentVerified": False,
        "historicalPublicTimesIndependentlyVerified": 0,
        "trainingReady": False,
        "predictionReady": False,
        "promotionAllowed": False,
        "productionDbModified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Private SEC claim-to-source review packet")
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if len(args.report) > MAX_CASES:
            raise ValueError("SEC_CLAIM_REVIEW_REPORTS_INVALID")
        primary, digest = _read_json(args.primary, MAX_INPUT)
        reports = [_read_json(path, MAX_REPORT_BYTES) for path in args.report]
        result = build_claim_review(primary, digest, args.source_dir, reports)
        _write(args.output, result)
        print(json.dumps({
            "status": result["status"],
            "casesIncluded": result["casesIncluded"],
            "claimCount": result["claimCount"],
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
            else "SEC_CLAIM_REVIEW_INVALID_OR_UNSAFE_INPUT"
        )
        parser.exit(2, f"sec-claim-review: ERROR_CODE={code}\n")


if __name__ == "__main__":
    main()
