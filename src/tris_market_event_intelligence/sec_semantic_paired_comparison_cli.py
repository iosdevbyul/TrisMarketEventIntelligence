"""Offline same-SEC-case paired comparison with no invented semantic quality score.

Independently verify both historical reports against the SAME SHA-pinned source.
Output contains only paths/counts: no original SEC text or model prose.
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


def _fields(case: dict) -> dict[str, str]:
    fields: dict[str, str] = {}
    if case.get("analysisStatus") == "INSUFFICIENT_SUBSTANTIVE_EVIDENCE":
        return {"analysisStatus": case["analysisStatus"], "summary": case["summary"]}
    for i, observation in enumerate(case["observations"]):
        fields[f"observations[{i}].interpretation"] = observation["interpretation"]
    for i, hypothesis in enumerate(case["hypotheses"]):
        category = hypothesis["category"]
        fields[f"hypotheses[{category}].mechanism"] = hypothesis["mechanism"]
        fields[f"hypotheses[{category}].limitation"] = hypothesis["limitation"]
    fields["summary"] = case["summary"]
    return fields


def compare_semantic_reports(
    primary: dict,
    primary_sha: str,
    source_dir: Path,
    baseline: list[tuple[dict, str]],
    candidate: list[tuple[dict, str]],
) -> dict:
    """Reject unmatched originals; differences are not correctness judgments."""
    base = build_review_packet(primary, primary_sha, source_dir, baseline)
    new = build_review_packet(primary, primary_sha, source_dir, candidate)
    a = {c["caseIndex"]: c for c in base["cases"]}
    b = {c["caseIndex"]: c for c in new["cases"]}
    if not a or a.keys() != b.keys():
        raise ValueError("SEC_PAIRED_CASE_SET_MISMATCH")
    pairs: list[dict] = []
    total_fields = changed_fields = 0
    for index in sorted(a):
        old, updated = a[index], b[index]
        if (old["accession"] != updated["accession"]
                or old["sourceSha256"] != updated["sourceSha256"]):
            raise ValueError("SEC_PAIRED_SOURCE_IDENTITY_MISMATCH")
        if old["semanticReportSha256"] == updated["semanticReportSha256"]:
            raise ValueError("SEC_PAIRED_INDEPENDENT_CANDIDATE_REQUIRED")
        old_fields, new_fields = _fields(old), _fields(updated)
        if old.get("analysisStatus") != updated.get("analysisStatus"):
            old_fields["analysisStatus"] = old.get("analysisStatus", "LEGACY_ANALYZED")
            new_fields["analysisStatus"] = updated.get("analysisStatus", "LEGACY_ANALYZED")
        # Observation count can change between generations, but changes must
        # be explicit and never silently counted as no change.
        locations = sorted(old_fields.keys() | new_fields.keys())
        deltas = [{
            "location": loc,
            "change": (
                "ADDED" if loc not in old_fields else
                "REMOVED" if loc not in new_fields else
                "UNCHANGED" if old_fields[loc] == new_fields[loc] else
                "CHANGED"
            ),
        } for loc in locations]
        total_fields += len(deltas)
        changed_fields += sum(d["change"] != "UNCHANGED" for d in deltas)
        pairs.append({
            "caseIndex": index,
            "evidenceScopeChanged": old.get("sourceSpan") != updated.get("sourceSpan"),
            "baselineSourceSpanSha256": old.get("sourceSpan", {}).get("spanSha256"),
            "candidateSourceSpanSha256": updated.get("sourceSpan", {}).get("spanSha256"),
            "baselineSemanticReportSha256": old["semanticReportSha256"],
            "candidateSemanticReportSha256": updated["semanticReportSha256"],
            "fields": deltas,
        })
    return {
        "schemaVersion": "sec_primary_semantic_paired_comparison_v1",
        "status": "DIFFERENCE_ONLY_HUMAN_REVIEW_REQUIRED",
        "primaryReportSha256": primary_sha,
        "pairedCases": len(pairs),
        "fieldsCompared": total_fields,
        "fieldsDifferent": changed_fields,
        "pairs": pairs,
        "semanticQualityImprovementVerified": False,
        "humanDecisionsRecorded": 0,
        "historicalPublicTimesIndependentlyVerified": 0,
        "trainingReady": False,
        "predictionReady": False,
        "promotionAllowed": False,
        "productionDbModified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline SEC baseline/candidate comparison")
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, action="append", required=True)
    parser.add_argument("--candidate-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if len(args.baseline_report) > MAX_CASES or len(args.candidate_report) > MAX_CASES:
            raise ValueError("SEC_PAIRED_REPORT_COUNT_INVALID")
        primary, sha = _read_json(args.primary, MAX_INPUT)
        baseline = [_read_json(path, MAX_REPORT_BYTES) for path in args.baseline_report]
        candidate = [_read_json(path, MAX_REPORT_BYTES) for path in args.candidate_report]
        result = compare_semantic_reports(primary, sha, args.source_dir, baseline, candidate)
        _write(args.output, result)
        print(json.dumps({
            "status": result["status"],
            "pairedCases": result["pairedCases"],
            "fieldsDifferent": result["fieldsDifferent"],
            "semanticQualityImprovementVerified": False,
            "humanDecisionsRecorded": 0,
            "outputCreated": True,
        }))
    except (OSError, ValueError, KeyError, TypeError, UnicodeError,
            json.JSONDecodeError, AttributeError) as exc:
        code = (
            exc.args[0] if type(exc) in {ValueError, TypeError}
            and exc.args and isinstance(exc.args[0], str)
            and re.fullmatch(r"SEC_[A-Z0-9_]{8,90}", exc.args[0])
            else "SEC_PAIRED_INVALID_OR_UNSAFE_INPUT"
        )
        parser.exit(2, f"sec-semantic-compare: ERROR_CODE={code}\n")


if __name__ == "__main__":
    main()
