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
from tris_market_event_intelligence.sec_source_lineage_audit_cli import _read_json
from tris_market_event_intelligence.sec_semantic_evidence_gate import (
    INSUFFICIENT,
    source_evidence_status,
)


def prepare_candidate_replay(
    primary: dict, primary_sha: str, source_dir: Path,
    baseline: list[tuple[dict, str]], *, start_index: int, limit: int,
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
    return prepared, {
        "schemaVersion": "sec_semantic_controlled_candidate_plan_v1",
        "status": "VERIFIED_BASELINE_REPLAY_READY",
        "primaryReportSha256": primary_sha,
        "startIndex": start_index,
        "caseIndexes": sorted(wanted),
        "baselineCasesValidated": reviewed["reviewCasesPrepared"],
        "sourceQuotesReverified": True,
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
        )
        if args.prepare:
            print(json.dumps(plan))
            return
        stage = "INITIALIZE_PROVIDER"
        requested = prepared["cases"][
            plan["startIndex"]:plan["startIndex"] + len(plan["caseIndexes"])
        ]
        if all(
            source_evidence_status(case["evidence"][0]["text"]) == INSUFFICIENT
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
