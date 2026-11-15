"""Generate URL-only historical review packets; optionally run injected Gemma."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from tris_market_event_intelligence.historical_story_review import (
    prepare_historical_story_review,
    review_story_with_model,
)
from tris_market_event_intelligence.llm_runtime import LLMSettings
from tris_market_event_intelligence.news_llm_factory import build_runtime


def _save_private(path: Path, result: dict[str, object]) -> None:
    if (path.is_symlink() or path.exists() or not path.parent.is_dir()
            or path.parent.is_symlink()):
        raise ValueError("Only a new private output path is permitted")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Source-only historical NEWS story candidates and opt-in Gemma"
    )
    parser.add_argument("--scout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--limit", type=int, default=12)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    try:
        packet = prepare_historical_story_review(args.scout)
        if args.run:
            settings = replace(
                LLMSettings.from_environment(explicit=True),
                model_id=args.model,
            )
            runtime = build_runtime(settings, confirm_model=True)
            result = review_story_with_model(packet, provider=runtime, limit=args.limit)
        else:
            result = packet
        _save_private(args.output, result)
        print(json.dumps({
            "status": result["status"],
            "shockDate": packet["shockDate"],
            "referenceCount": packet["referenceCount"],
            "candidateRepostGroups": packet["candidateRepostGroupCount"],
            "candidateCategoryCounts": packet["candidateCategoryCounts"],
            "officialPrimarySource": bool(packet["officialPrimarySource"]),
            "modelCalls": result["modelCalls"],
            "trainingReady": False,
            "outputCreated": True,
        }, ensure_ascii=False))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        # Never log model response, source URL, local private path or secrets.
        code = ("STORY_REVIEW_FAILED" if not args.run else
                "STORY_MODEL_REVIEW_FAILED")
        parser.exit(2, f"historical-story-review: {code} ({type(exc).__name__})\n")


if __name__ == "__main__":
    main()
