"""Opt-in full archived NEWS candidate set to learned-method Gemma research."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from tris_market_event_intelligence.historical_multi_hypothesis import (
    build_research_context,
    read_reviewed_lessons,
    research_from_scout,
)
from tris_market_event_intelligence.historical_story_review import (
    prepare_historical_story_review,
)
from tris_market_event_intelligence.llm_runtime import LLMExecutionError, LLMSettings
from tris_market_event_intelligence.news_llm_factory import build_runtime


def main() -> None:
    parser = argparse.ArgumentParser(
        description="All-source evidence, three competing Gemma hypotheses"
    )
    parser.add_argument("--scout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--reviewed-lessons", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    try:
        if (args.output.is_symlink() or args.output.exists()
                or not args.output.parent.is_dir()
                or args.output.parent.is_symlink()):
            raise ValueError("Research artifact must be a new private file")
        story = prepare_historical_story_review(args.scout)
        lessons, sha = read_reviewed_lessons(args.reviewed_lessons)
        if args.preview:
            context = build_research_context(story=story, lessons=lessons)
            result = {
                "schemaVersion": "historical_multi_hypothesis_preflight_v1",
                "status": "PREVIEW_ONLY",
                "allCandidateReferenceCount": context["allReferencesProvided"],
                "primarySourcePresented": context[
                    "firstPartyDisclosureSeparateFromUrlHints"
                ] is not None,
                "usedHumanReviewedLessons": bool(lessons),
                "humanReviewedLessonsSha256": sha,
                "modelCalls": 0, "weightTrainingPerformed": False,
                "promotionAllowed": False,
            }
        else:
            settings = replace(
                LLMSettings.from_environment(explicit=True), model_id=args.model,
            )
            runtime = build_runtime(settings, confirm_model=True)
            result = research_from_scout(
                args.scout, provider=runtime,
                reviewed_lessons=args.reviewed_lessons,
            )
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(args.output, flags, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
        except BaseException:
            args.output.unlink(missing_ok=True)
            raise
        print(json.dumps({
            "status": result["status"],
            "newsReferencesConsidered": result["allCandidateReferenceCount"],
            "primarySourcePresented": result["primarySourcePresented"],
            "reviewedLessonsUsed": result["usedHumanReviewedLessons"],
            "hypotheses": len(result.get("hypotheses", [])),
            "modelCalls": result["modelCalls"],
            "trainingWeightsChanged": False,
            "predictionReady": False,
            "outputCreated": True,
        }, ensure_ascii=False))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        # Stable diagnostic codes only, never original source/model text or keys.
        if isinstance(exc, LLMExecutionError):
            reason = exc.code
        elif isinstance(exc, json.JSONDecodeError):
            reason = "MODEL_JSON_INVALID"
        elif isinstance(exc, ValueError):
            reason = {
                "Bounded whole-cohort input exceeded resource cap": "MODEL_INPUT_OVERSIZED",
                "Require three competing hypotheses": "MODEL_HYPOTHESES_COUNT_INVALID",
                "Oversized model response": "MODEL_OUTPUT_OVERSIZED",
                "Research model contract violated": "MODEL_OUTPUT_SCHEMA_INVALID",
                "Malformed candidate hypothesis": "MODEL_HYPOTHESIS_SCHEMA_INVALID",
                "Unsupported source reference or evidence class": "MODEL_SOURCE_PROVENANCE_INVALID",
                "Primary evidence not cited": "MODEL_PRIMARY_EVIDENCE_MISSING",
                "Model cannot certify causality or forecast skill": "MODEL_UNSAFE_CLAIM_REJECTED",
                "Missing substantive research caveats": "MODEL_EXPLANATION_INCOMPLETE",
                "Model omitted research limitation": "MODEL_LIMITATION_INCOMPLETE",
                "Research artifact must be a new private file": "OUTPUT_ALREADY_EXISTS_OR_UNSAFE",
            }.get(str(exc), "SOURCE_OR_CONFIG_VALIDATION_FAILED")
        else:
            reason = "SOURCE_OR_OUTPUT_IO_INVALID"
        parser.exit(2, f"historical-multi-research: {reason}\\n")


if __name__ == "__main__":
    main()
