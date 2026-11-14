"""Local, create-only news-human-review preparation and scoring CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tris_market_event_intelligence.news_importance_review import (
    prepare_review,
    score_review,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare or score human-only news importance reviews"
    )
    parser.add_argument("--cohort", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--score", type=Path, metavar="HUMAN_REVIEWS_JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = (
            prepare_review(args.cohort) if args.prepare
            else score_review(args.cohort, args.score)
        )
        if args.output.is_symlink() or not args.output.parent.is_dir():
            raise ValueError("Existing ordinary output directory required")
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"news-human-importance: {type(exc).__name__}: {exc}\n")
    print(json.dumps({
        "status": result.get("status", "UNLABELLED_HUMAN_REVIEW_TEMPLATE"),
        "judgments": len(result["judgments"]) if args.prepare else result["humanReviewedArticles"],
        "trainingDatasetEligible": False,
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
