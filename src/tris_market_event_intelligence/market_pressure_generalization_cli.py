"""Cross-event generalization study on time-bounded collected news.

This is a prospective coverage / structural-quality audit, NOT an accuracy
claim or training run. No prompts specific to Nvidia, Pakistan, or article IDs.
Input DB is read-only; outputs are private and create-only.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

from tris_market_event_intelligence.llm_runtime import (
    LLMCapabilities,
    LLMSettings,
)
from tris_market_event_intelligence.market_event_discovery import (
    discover_and_analyze,
    recent_candidates,
)
from tris_market_event_intelligence.market_pressure_trace import (
    MarketPressureTrace,
    TracedMarketProvider,
)
from tris_market_event_intelligence.news_llm_factory import build_runtime


def evaluate_pages(
    db: Path, *, as_of: date, lookback_days: int, pages: int,
    provider: object | None,
) -> dict:
    if not 1 <= pages <= 8:
        raise ValueError("GENERALIZATION_PAGE_LIMIT_INVALID")
    if not 1 <= lookback_days <= 120:
        raise ValueError("GENERALIZATION_LOOKBACK_INVALID")
    summaries = []
    seen: set[str] = set()
    total_calls = 0
    group_count = 0
    proposed = 0
    accepted = 0
    rejected = 0
    source_kinds: dict[str, int] = {}
    source_publishers: set[str] = set()
    publisher_counts: dict[str, int] = {}
    observed_group_sources: set[str] = set()
    for page in range(pages):
        candidates, inventory = recent_candidates(
            db, as_of=as_of, lookback_days=lookback_days, page=page,
            sampling_mode="source_balanced",
        )
        source_ids = {row["evidence"].sourceId for row in candidates}
        overlap = seen.intersection(source_ids)
        if overlap:
            raise ValueError("GENERALIZATION_CROSS_PAGE_SOURCE_OVERLAP")
        seen.update(source_ids)
        for row in candidates:
            source = row["evidence"]
            source_kinds[source.kind] = source_kinds.get(source.kind, 0) + 1
            source_publishers.add(source.sourceName)
            publisher_counts[source.sourceName] = (
                publisher_counts.get(source.sourceName, 0) + 1
            )
        group_summaries = []
        if provider is not None and candidates:
            result = discover_and_analyze(
                db, as_of=as_of, lookback_days=lookback_days,
                page=page, max_groups=2, reasoning_mode="causal",
                sampling_mode="source_balanced", provider=provider,
            )
            total_calls += result["modelCalls"]
            for event in result["events"]:
                a = event["analysis"]
                observed_group_sources.update(event["groupSourceIds"])
                group_count += 1
                proposed += a.get("proposedImpactCount", 0)
                accepted += len(a["impacts"])
                rejected += len(a.get("rejectedImpacts", []))
                group_summaries.append({
                    "sourceIds": event["groupSourceIds"],
                    "status": a["status"],
                    "proposedImpactCount": a.get("proposedImpactCount", 0),
                    "acceptedImpactCount": len(a["impacts"]),
                    "rejectedImpactCount": len(a.get("rejectedImpacts", [])),
                    "pathCount": sum(
                        len(item.get("causalPaths", []))
                        for item in a["impacts"]
                    ),
                    "claimCount": len(a.get("groundedClaims", [])),
                    "directionCounts": {
                        key: sum(
                            item["direction"] == key for item in a["impacts"]
                        )
                        for key in ("POSITIVE", "NEGATIVE", "MIXED")
                    },
                    "strengthUnassessed": all(
                        item["strength"] == "UNASSESSED"
                        for item in a["impacts"]
                    ),
                    "errors": a.get("rejectedImpacts", []),
                    # Preserve model-authored conditional paths for actual
                    # human semantic comparison across previously unseen
                    # events. These are local-only private research outputs.
                    "causalResearchItems": [
                        {
                            "proposalIndex": item.get("proposalIndex"),
                            "targetType": item["targetType"],
                            "targetName": item["targetName"],
                            "direction": item["direction"],
                            "strength": item["strength"],
                            "causalPaths": item["causalPaths"],
                            "missingEvidence": item["missingEvidence"],
                            "uncertaintyExplanation": (
                                item["uncertaintyExplanation"]
                            ),
                            "supportingSourceIds": (
                                item["supportingSourceIds"]
                            ),
                            "economicCausalityVerified": False,
                        }
                        for item in a["impacts"]
                    ],
                    "groundedClaims": a.get("groundedClaims", []),
                })
        summaries.append({
            "page": page,
            "candidateCount": len(candidates),
            "hasMoreCandidatePages": inventory["hasMoreCandidatePages"],
            "sourceIds": sorted(source_ids),
            "groups": group_summaries,
        })
        if not inventory["hasMoreCandidatePages"]:
            break
    return {
        "schemaVersion": "general_causal_cross_event_evaluation_v1",
        "status": (
            "STRUCTURAL_RESEARCH_REQUIRES_HUMAN_REVIEW"
            if provider is not None else "READ_ONLY_COVERAGE_PREVIEW"
        ),
        "asOfDateExclusive": as_of.isoformat(),
        "lookbackDays": lookback_days,
        "candidatePagesExamined": len(summaries),
        "uniqueEligibleSources": len(seen),
        "sourceKindCounts": source_kinds,
        "samplingMode": "source_balanced",
        "distinctPublishers": len(source_publishers),
        "publisherCounts": dict(sorted(publisher_counts.items())),
        "candidateSourcesActuallyGrouped": len(observed_group_sources),
        "groupingCoverageFraction": (
            round(len(observed_group_sources) / len(seen), 4)
            if seen else None
        ),
        "samplingWarnings": [
            *(
                ["SOURCE_KIND_DIVERSITY_NOT_ESTABLISHED"]
                if len(source_kinds) < 2 else []
            ),
            *(
                ["PUBLISHER_DIVERSITY_NOT_ESTABLISHED"]
                if len(source_publishers) < 3 else []
            ),
            *(
                ["MODEL_ANALYZED_ONLY_A_SUBSET_OF_CANDIDATES"]
                if provider is not None and len(observed_group_sources) < len(seen)
                else []
            ),
        ],
        "modelCalls": total_calls,
        "eventGroupCount": group_count,
        "proposedImpactCount": proposed,
        "sourceValidatedImpactCount": accepted,
        "rejectedImpactCount": rejected,
        "semanticAccuracyVerified": False,
        "generalizationVerified": False,
        "holdoutIndependenceCertified": False,
        "marketReturnsAnalyzed": False,
        "databaseModified": False,
        "pages": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--pages", type=int, default=4)
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--trace", type=Path,
        help="Private create-only JSONL trace (default OUTPUT.trace.jsonl)",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.run and args.output is None:
        parser.error("--run requires a new --output path")
    if args.output is not None and (
        args.output.exists() or args.output.is_symlink()
    ):
        parser.error("GENERALIZATION_OUTPUT_ALREADY_EXISTS")
    if args.preview and args.trace is not None:
        parser.error("--trace requires --run")
    trace_path = (
        args.trace or Path(str(args.output) + ".trace.jsonl")
        if args.run else None
    )
    if trace_path is not None and (
        trace_path.exists() or trace_path.is_symlink()
    ):
        parser.error("GENERALIZATION_PRIVATE_TRACE_ALREADY_EXISTS")
    provider = None
    trace = None
    try:
        if args.run:
            settings = replace(
                LLMSettings.from_environment(explicit=True),
                model_id=args.model,
                capabilities=LLMCapabilities(
                    json_schema=True, thinking=False,
                    context_window=8192, max_output_tokens=4096,
                ),
            )
            # A failed model call still leaves a private, durable and
            # create-only trace, preventing repetitive blind reruns.
            if trace_path is None:
                raise ValueError("GENERALIZATION_TRACE_PATH_REQUIRED")
            trace = MarketPressureTrace(trace_path)
            trace.record(
                "GENERALIZATION_STARTED", cutoff=args.as_of.isoformat(),
                pages=args.pages, lookbackDays=args.lookback_days,
                samplingMode="source_balanced",
            )
            provider = TracedMarketProvider(
                build_runtime(settings, confirm_model=True), trace,
            )
        report = evaluate_pages(
            args.db, as_of=args.as_of, lookback_days=args.lookback_days,
            pages=args.pages, provider=provider,
        )
        if args.output:
            with args.output.open("x", encoding="utf-8") as out:
                json.dump(report, out, ensure_ascii=False, indent=2)
                out.write("\n")
        if trace is not None:
            trace.record(
                "GENERALIZATION_FINISHED", modelCalls=report["modelCalls"],
                sampledSourceCount=report["uniqueEligibleSources"],
                analyzedGroupCount=report["eventGroupCount"],
                structuralAcceptedCount=report["sourceValidatedImpactCount"],
                structuralRejectedCount=report["rejectedImpactCount"],
            )
    except Exception as error:
        if trace is not None:
            trace.validation_failure(
                stage="GENERALIZATION_RUN", error=error,
            )
        raise
    finally:
        if trace is not None:
            trace.close()
    print(json.dumps({
        "status": report["status"],
        "candidatePagesExamined": report["candidatePagesExamined"],
        "uniqueEligibleSources": report["uniqueEligibleSources"],
        "sourceKindCounts": report["sourceKindCounts"],
        "samplingMode": report["samplingMode"],
        "distinctPublishers": report["distinctPublishers"],
        "samplingWarnings": report["samplingWarnings"],
        "candidateSourcesActuallyGrouped": (
            report["candidateSourcesActuallyGrouped"]
        ),
        "groupingCoverageFraction": report["groupingCoverageFraction"],
        "eventGroupCount": report["eventGroupCount"],
        "proposedImpactCount": report["proposedImpactCount"],
        "sourceValidatedImpactCount": report["sourceValidatedImpactCount"],
        "rejectedImpactCount": report["rejectedImpactCount"],
        "modelCalls": report["modelCalls"],
        "generalizationVerified": False,
        "dbWrites": 0, "outputCreated": bool(args.output),
        "privateTracePath": str(trace_path) if trace_path else None,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
