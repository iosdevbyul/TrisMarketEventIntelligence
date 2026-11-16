"""Controlled create-only Gemma replay for exactly matched archived SEC cases.

Original raw SEC provenance, prior baseline and run target must all match.
No baseline overwrite, no external fetches, no price data and no auto approvals.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from pathlib import Path

from tris_market_event_intelligence.historical_issuer_corroboration_cli import _write
from tris_market_event_intelligence.llm_runtime import (
    LLMCapabilities,
    LLMExecutionError,
    LLMProvider,
    LLMSettings,
)
from tris_market_event_intelligence.news_llm_factory import build_runtime
from tris_market_event_intelligence.sec_pinned_body_spans import apply_body_spans
from tris_market_event_intelligence.sec_primary_semantic_research_cli import (
    MAX_CASES,
    MAX_INPUT,
    MAX_MODEL_CASES,
    prepare_primary_cases,
    run_semantic_research,
)
from tris_market_event_intelligence.sec_primary_semantic_review_cli import (
    MAX_REPORT_BYTES,
    build_review_packet,
)
from tris_market_event_intelligence.sec_ranked_multi_passages import (
    apply_ranked_passages,
)
from tris_market_event_intelligence.sec_revenue_context_guard import STOP_STATUSES
from tris_market_event_intelligence.sec_semantic_evidence_gate import (
    INSUFFICIENT,
    source_evidence_status,
)
from tris_market_event_intelligence.sec_source_lineage_audit_cli import _read_json


def prepare_candidate_replay(
    primary: dict, primary_sha: str, source_dir: Path,
    baseline: list[tuple[dict, str]], *, start_index: int, limit: int,
    body_spans: bool = False, multi_passages: bool = False,
) -> tuple[dict, dict]:
    """Fail closed unless every requested case exists in the verified baseline."""
    prepared = prepare_primary_cases(primary, primary_sha, source_dir)
    reviewed = build_review_packet(primary, primary_sha, source_dir, baseline)
    if (not isinstance(start_index, int) or isinstance(start_index, bool)
            or not isinstance(limit, int) or isinstance(limit, bool)
            or not 1 <= limit <= MAX_MODEL_CASES
            or not 0 <= start_index
            or start_index + limit > len(prepared["cases"])):
        raise ValueError("SEC_CANDIDATE_RANGE_INVALID")
    wanted = set(range(start_index, start_index + limit))
    present = {case["caseIndex"] for case in reviewed["cases"]}
    if not wanted <= present:
        raise ValueError("SEC_CANDIDATE_BASELINE_CASES_MISSING")
    if body_spans and multi_passages:
        raise ValueError("SEC_CANDIDATE_SELECTION_MODE_CONFLICT")
    if body_spans:
        prepared = apply_body_spans(prepared, source_dir)
    elif multi_passages:
        prepared = apply_ranked_passages(prepared, source_dir)
    return prepared, {
        "schemaVersion": "sec_semantic_controlled_candidate_plan_v1",
        "status": "VERIFIED_BASELINE_REPLAY_READY",
        "primaryReportSha256": primary_sha,
        "startIndex": start_index,
        "caseIndexes": sorted(wanted),
        "baselineCasesValidated": reviewed["reviewCasesPrepared"],
        "sourceQuotesReverified": True,
        "bodySpansEnabled": body_spans,
        "multiPassagesEnabled": multi_passages,
        "multiPassageCasesSelected": sum(
            "sourcePassages" in prepared["cases"][index] for index in sorted(wanted)
        ),
        "selectedPassagesCount": sum(
            len(prepared["cases"][index].get("sourcePassages", []))
            for index in sorted(wanted)
        ),
        "requestedBodySpansSelected": sum(
            "sourceSpan" in prepared["cases"][index] for index in sorted(wanted)
        ),
        "referentsUnresolved": sum(
            prepared["cases"][index].get("sourceSpan", {}).get("contextStatus")
            == "REFERENT_UNRESOLVED" for index in sorted(wanted)
        ),
        "descriptiveOnly": sum(
            prepared["cases"][index].get("sourceSpan", {}).get("contextStatus")
            == "DESCRIPTIVE_REVENUE_SHARE_ONLY" for index in sorted(wanted)
        ),
        "historicalPublicTimesIndependentlyVerified": 0,
        "humanDecisionsRecorded": 0,
        "qualityImprovementVerified": False,
        "trainingReady": False,
        "promotionAllowed": False,
    }


def replay_candidate(
    prepared: dict, plan: dict, provider: LLMProvider,
) -> dict:
    report = run_semantic_research(
        prepared, provider, start_index=plan["startIndex"],
        limit=len(plan["caseIndexes"]), evidence_gated=True,
    )
    if ([case["caseIndex"] for case in report["cases"]] != plan["caseIndexes"]
            or report["primaryReportSha256"] != plan["primaryReportSha256"]
            or report["trainingReady"] is not False
            or report["promotionAllowed"] is not False):
        raise ValueError("SEC_CANDIDATE_REPLAY_IDENTITY_INVALID")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Controlled SEC semantic candidate replay")
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, action="append", required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--limit", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--model", default="gemma4:e2b")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--body-spans", action="store_true")
    selection.add_argument("--multi-passages", action="store_true")
    parser.add_argument("--output", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    stage = "VERIFY_ORIGINAL_AND_BASELINE"
    try:
        if len(args.baseline_report) > MAX_CASES:
            raise ValueError("SEC_CANDIDATE_BASELINE_REPORT_COUNT_INVALID")
        if args.run:
            if args.output is None:
                raise ValueError("SEC_CANDIDATE_NEW_OUTPUT_REQUIRED")
            if (args.output.is_symlink() or args.output.exists()
                    or not args.output.parent.is_dir()
                    or args.output.parent.is_symlink()):
                raise ValueError("SEC_CANDIDATE_OUTPUT_NOT_CREATE_ONLY")
        primary, sha = _read_json(args.primary, MAX_INPUT)
        baseline = [_read_json(path, MAX_REPORT_BYTES) for path in args.baseline_report]
        prepared, plan = prepare_candidate_replay(
            primary, sha, args.source_dir, baseline,
            start_index=args.start_index, limit=args.limit,
            body_spans=args.body_spans,
            multi_passages=args.multi_passages,
        )
        if args.prepare:
            print(json.dumps(plan))
            return
        stage = "INITIALIZE_PROVIDER"
        requested = prepared["cases"][
            plan["startIndex"]:plan["startIndex"] + len(plan["caseIndexes"])
        ]
        if all(
            (
                (INSUFFICIENT if args.multi_passages
                 and "sourcePassages" not in case else None)
                or case.get("sourceSpan", {}).get("contextStatus")
                or (INSUFFICIENT if source_evidence_status(
                    case["evidence"][0]["text"]
                ) == INSUFFICIENT else None)
            ) in STOP_STATUSES
            for case in requested
        ):
            # No Ollama probing or provider network when every case abstains.
            class NoModelProvider:
                model_id = args.model

                def generate_json(self, *, system: str, user: str, schema: dict) -> str:
                    raise AssertionError("SEC_MODEL_MUST_NOT_RUN_ON_INSUFFICIENT_SOURCE")

            provider = NoModelProvider()
        else:
            settings = replace(
                LLMSettings.from_environment(explicit=True), model_id=args.model,
                capabilities=LLMCapabilities(
                    json_schema=True, thinking=False,
                    context_window=8192, max_output_tokens=3072,
                ),
            )
            provider = build_runtime(settings, confirm_model=True)
        stage = "GENERATE_AND_VALIDATE_CANDIDATE"
        candidate = replay_candidate(prepared, plan, provider)
        stage = "WRITE_CREATE_ONLY_PRIVATE_OUTPUT"
        _write(args.output, candidate)
        print(json.dumps({
            "status": candidate["status"],
            "casesAnalyzed": candidate["casesAnalyzed"],
            "casesAbstained": candidate["casesAbstained"],
            "bodySpansSelected": plan["requestedBodySpansSelected"],
            "multiPassageCasesSelected": plan["multiPassageCasesSelected"],
            "selectedPassagesCount": plan["selectedPassagesCount"],
            "referentsUnresolved": plan["referentsUnresolved"],
            "descriptiveOnly": plan["descriptiveOnly"],
            "caseIndexes": plan["caseIndexes"],
            "modelCalls": candidate["modelCalls"],
            "baselineModified": False,
            "humanDecisionsRecorded": 0,
            "qualityImprovementVerified": False,
            "trainingReady": False,
            "outputCreated": True,
        }))
    except (OSError, ValueError, KeyError, TypeError, UnicodeError,
            json.JSONDecodeError, AttributeError) as exc:
        if isinstance(exc, LLMExecutionError):
            code = exc.code
        elif (type(exc) in {ValueError, TypeError}
              and exc.args and isinstance(exc.args[0], str)
              and re.fullmatch(r"SEC_[A-Z0-9_]{8,90}", exc.args[0])):
            code = exc.args[0]
        else:
            code = "SEC_CANDIDATE_REPLAY_INVALID_OR_UNSAFE_INPUT"
        parser.exit(2, f"sec-candidate-replay: FAILED_STAGE={stage} ERROR_CODE={code}\n")


if __name__ == "__main__":
    main()
