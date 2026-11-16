"""Offline, SHA-backed human-review packet for bounded SEC semantic case reports.

Never fetches a source, calls a model, reads prices, or promotes hypotheses.
Reports stay in private local files with create-only permissions.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tris_market_event_intelligence.historical_issuer_corroboration_cli import _write
from tris_market_event_intelligence.sec_pinned_body_spans import (
    CONTEXT_VERSION,
    VERSION,
    reverify_body_span,
    select_versioned_body_span,
)
from tris_market_event_intelligence.sec_primary_semantic_research_cli import (
    MAX_CASES,
    MAX_INPUT,
    _validate_response,
    prepare_primary_cases,
)
from tris_market_event_intelligence.sec_ranked_multi_passages import (
    VERSION as MULTI_VERSION,
    reverify_ranked_passages,
    select_ranked_passages,
)
from tris_market_event_intelligence.sec_revenue_context_guard import (
    STOP_STATUSES,
    validated_stopped_interpretation,
)
from tris_market_event_intelligence.sec_semantic_evidence_gate import (
    ABSTENTION_SUMMARY,
    INSUFFICIENT,
    SUBSTANTIVE,
    source_evidence_status,
)
from tris_market_event_intelligence.sec_source_lineage_audit_cli import _read_json

MAX_REPORT_BYTES = 900_000
SHA = re.compile(r"[a-f0-9]{64}\Z")


def build_review_packet(
    primary: dict, primary_sha: str, source_dir: Path,
    reports: list[tuple[dict, str]],
) -> dict:
    """Recheck exact local source bytes and all model claims before human review."""
    prepared = prepare_primary_cases(primary, primary_sha, source_dir)
    cases = prepared["cases"]
    by_accession = {c["accession"]: (i, c) for i, c in enumerate(cases)}
    if not reports or len(reports) > MAX_CASES:
        raise ValueError("SEC_SEMANTIC_REVIEW_REPORTS_INVALID")
    seen: set[str] = set()
    entries: list[dict] = []
    model_calls = 0
    for report, digest in reports:
        if (not isinstance(report, dict) or not SHA.fullmatch(digest)
                or report.get("schemaVersion") != "sec_primary_source_semantic_research_v1"
                or report.get("status") != "SOURCE_QUOTED_LLM_HYPOTHESES_HUMAN_REVIEW_REQUIRED"
                or report.get("primaryReportSha256") != primary_sha
                or report.get("trainingReady") is not False
                or report.get("predictionReady") is not False
                or report.get("promotionAllowed") is not False
                or report.get("stockOutcomesConsumed") is not False
                or report.get("productionDbModified") is not False
                or report.get("allQuoteSpansVerifiedAgainstOriginalSHA") is not True
                or report.get("historicalPublicTimesIndependentlyVerified") != 0
                or report.get("verifiedCausalEffects") != 0
                or not isinstance(report.get("cases"), list)
                or not 1 <= len(report["cases"]) <= 3
                or report.get("casesAnalyzed") != len(report["cases"])
                or not isinstance(report.get("modelId"), str)
                or not report["modelId"]):
            raise ValueError("SEC_SEMANTIC_REVIEW_REPORT_PROVENANCE_INVALID")
        body_mode = report.get("bodySpanSelectionVersion")
        multi_mode = report.get("multiPassageVersion")
        if multi_mode is not None and (
            multi_mode != MULTI_VERSION or body_mode is not None
        ):
            raise ValueError("SEC_MULTI_REPORT_VERSION_INVALID")
        if body_mode is not None and body_mode not in {VERSION, CONTEXT_VERSION}:
            raise ValueError("SEC_BODY_SPAN_SELECTION_VERSION_INVALID")
        corrections = 0
        abstained = 0
        for item in report["cases"]:
            if not isinstance(item, dict):
                raise TypeError("SEC_SEMANTIC_REVIEW_CASE_INVALID")
            acc = item.get("accession")
            if not isinstance(acc, str) or acc not in by_accession or acc in seen:
                raise ValueError("SEC_SEMANTIC_REVIEW_DUPLICATE_OR_UNKNOWN_CASE")
            index, case = by_accession[acc]
            if multi_mode == MULTI_VERSION:
                raw_path = source_dir / f"sec-{acc}-primary.source"
                if (raw_path.is_symlink() or not raw_path.is_file()
                        or raw_path.stat().st_size > 8_000_000):
                    raise ValueError("SEC_MULTI_ORIGINAL_SOURCE_REQUIRED")
                raw = raw_path.read_bytes()
                expected_passages = select_ranked_passages(case, raw)
                if ("sourcePassages" in item) != (len(expected_passages) >= 2):
                    raise ValueError("SEC_MULTI_SELECTION_MISMATCH")
                if len(expected_passages) >= 2:
                    case = reverify_ranked_passages(
                        case, raw, item["sourcePassages"],
                    )
            elif "sourcePassages" in item:
                raise ValueError("SEC_MULTI_REPORT_VERSION_REQUIRED")
            if body_mode in {VERSION, CONTEXT_VERSION}:
                raw_path = source_dir / f"sec-{acc}-primary.source"
                if raw_path.is_symlink() or not raw_path.is_file():
                    raise ValueError("SEC_BODY_ORIGINAL_SOURCE_REQUIRED")
                raw = raw_path.read_bytes()
                expected = select_versioned_body_span(case, raw, body_mode)
                if ("sourceSpan" in item) != (expected is not None):
                    raise ValueError("SEC_BODY_SPAN_SELECTION_MISMATCH")
                if expected is not None:
                    case = reverify_body_span(
                        case, raw, item["sourceSpan"], version=body_mode,
                    )
            elif "sourceSpan" in item:
                raise ValueError("SEC_BODY_SPAN_MODE_REQUIRED")
            if (item.get("sourceSha256") != case["sourceSha256"]
                    or item.get("historicalTimeEligibility") is not False
                    or item.get("sourceInterpretationIsHumanUnverified") is not True
                    or not isinstance(item.get("hypothesesCorrectionAttempted"), bool)
                    or ("caseIndex" in item and item["caseIndex"] != index)):
                raise ValueError("SEC_SEMANTIC_REVIEW_CASE_PROVENANCE_INVALID")
            analysis_status = item.get("analysisStatus")
            gated = report.get("evidenceGateVersion") == "sec_substantive_evidence_v1"
            if gated and analysis_status not in STOP_STATUSES | {SUBSTANTIVE}:
                raise ValueError("SEC_SEMANTIC_ANALYSIS_STATUS_INVALID")
            if analysis_status in STOP_STATUSES and gated:
                expected_context = (
                    case.get("sourceSpan", {}).get("contextStatus")
                    if body_mode == CONTEXT_VERSION else None
                )
                expected_status = (
                    INSUFFICIENT if multi_mode == MULTI_VERSION else
                    expected_context
                    or (INSUFFICIENT if source_evidence_status(
                        case["evidence"][0]["text"]
                    ) == INSUFFICIENT else None)
                )
                expected_interpretation = (
                    {"observations": [], "hypotheses": [],
                     "summary": ABSTENTION_SUMMARY}
                    if analysis_status == INSUFFICIENT
                    else validated_stopped_interpretation(analysis_status)
                )
                if (analysis_status != expected_status
                        or item.get("quoteAnchorSelectionApplied") is not False
                        or item["hypothesesCorrectionAttempted"] is not False
                        or item.get("interpretation") != expected_interpretation):
                    raise ValueError("SEC_SEMANTIC_ABSTENTION_PROVENANCE_INVALID")
                validated = item["interpretation"]
                abstained += 1
            else:
                if analysis_status is not None and (
                    not gated or analysis_status != SUBSTANTIVE
                    or source_evidence_status(
                        case["evidence"][0].get("focusText",
                                               case["evidence"][0]["text"])
                    ) != SUBSTANTIVE
                ):
                    raise ValueError("SEC_SEMANTIC_ANALYSIS_STATUS_INVALID")
                validated = (
                    validate_ranked_interpretation(case, item.get("interpretation"))
                    if multi_mode == MULTI_VERSION else
                    _validate_response(
                        case, json.dumps(item.get("interpretation"), ensure_ascii=False),
                    )
                )
            seen.add(acc)
            corrections += int(item["hypothesesCorrectionAttempted"])
            entries.append({
                "caseIndex": index,
                **({"sourcePassages": case["sourcePassages"]}
                   if "sourcePassages" in case else {}),
                **({"sourceSpan": case["sourceSpan"]} if "sourceSpan" in case else {}),
                **({"analysisStatus": analysis_status} if gated else {}),
                "accession": acc,
                "form": case["form"],
                "filingDate": case["filingDate"],
                "sourceSha256": case["sourceSha256"],
                "sourceExcerpt": case["evidence"][0]["text"],
                **({"sourceExcerpts": [
                    {"evidenceId": evidence["id"], "text": evidence["text"],
                     "focusText": evidence["focusText"]}
                    for evidence in case["evidence"]
                ]} if "sourcePassages" in case else {}),
                "modelId": report["modelId"],
                "semanticReportSha256": digest,
                "observations": [
                    {
                        **observation,
                        "humanQuoteAccurate": None,
                        "humanInterpretationSupported": None,
                    }
                    for observation in validated["observations"]
                ],
                "hypotheses": [
                    {
                        **hypothesis,
                        "humanReasonablePossibility": None,
                        "humanDoesNotImplyCausality": None,
                    }
                    for hypothesis in validated["hypotheses"]
                ],
                "summary": validated["summary"],
                "humanSummarySupported": None,
                "humanNotes": "",
                "humanReviewComplete": False,
                "historicalPublicTimeVerified": False,
                "trainingEligible": False,
            })
        if report.get("hypothesesCorrectionCount") != corrections:
            raise ValueError("SEC_SEMANTIC_REVIEW_CORRECTION_COUNT_INVALID")
        if gated and report.get("casesAbstained") != abstained:
            raise ValueError("SEC_SEMANTIC_ABSTENTION_COUNT_INVALID")
        if (not gated and ("casesAbstained" in report or "evidenceGateVersion" in report)):
            raise ValueError("SEC_SEMANTIC_REVIEW_GATE_VERSION_INVALID")
        if report.get("modelCalls") != len(report["cases"]) - abstained + corrections:
            raise ValueError("SEC_SEMANTIC_REVIEW_MODEL_CALL_COUNT_INVALID")
        model_calls += report["modelCalls"]
    entries.sort(key=lambda row: row["caseIndex"])
    remaining = [
        {"caseIndex": i, "accession": c["accession"]}
        for i, c in enumerate(cases) if c["accession"] not in seen
    ]
    return {
        "schemaVersion": "sec_primary_semantic_human_review_packet_v1",
        "status": "HUMAN_REVIEW_PENDING",
        "primaryReportSha256": primary_sha,
        "totalPinnedCases": len(cases),
        "reviewCasesPrepared": len(entries),
        "pendingCaseIndexes": remaining,
        "modelCallsObserved": model_calls,
        "allSECQuotesReverifiedAgainstPinnedSources": True,
        "independentHistoricalPublicTimesVerified": 0,
        "trainingReady": False,
        "predictionReady": False,
        "promotionAllowed": False,
        "productionDbModified": False,
        "cases": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline local SEC-source recheck and private human-review packet"
    )
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage = "VALIDATE_REVIEW_INPUTS"
    try:
        if len(args.report) > MAX_CASES:
            raise ValueError("SEC_SEMANTIC_REVIEW_REPORTS_INVALID")
        primary, sha = _read_json(args.primary, MAX_INPUT)
        reports = [_read_json(path, MAX_REPORT_BYTES) for path in args.report]
        packet = build_review_packet(primary, sha, args.source_dir, reports)
        stage = "WRITE_PRIVATE_REVIEW_PACKET"
        _write(args.output, packet)
        print(json.dumps({
            "status": packet["status"],
            "reviewCasesPrepared": packet["reviewCasesPrepared"],
            "totalPinnedCases": packet["totalPinnedCases"],
            "casesRemaining": len(packet["pendingCaseIndexes"]),
            "allSECQuotesReverifiedAgainstPinnedSources": True,
            "trainingReady": False,
            "outputCreated": True,
        }))
    except (OSError, ValueError, KeyError, TypeError, UnicodeError,
            json.JSONDecodeError, AttributeError) as exc:
        code = (
            exc.args[0] if type(exc) in {ValueError, TypeError}
            and exc.args and isinstance(exc.args[0], str)
            and re.fullmatch(r"SEC_[A-Z0-9_]{8,90}", exc.args[0])
            else "SEC_SEMANTIC_REVIEW_INPUT_OR_WRITE_INVALID"
        )
        parser.exit(2, f"sec-semantic-review: FAILED_STAGE={stage} ERROR_CODE={code}\n")


if __name__ == "__main__":
    main()
