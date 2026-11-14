"""Offline research-only news importance probability evaluation CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tris_market_event_intelligence.news_importance_evaluation import (
    evaluate_importance_predictions,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare news importance predictions with human-only labels"
    )
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = evaluate_importance_predictions(
            cohort_path=args.cohort, review_path=args.review,
            predictions_path=args.predictions,
        )
        if args.output.is_symlink() or not args.output.parent.is_dir():
            raise ValueError("Existing ordinary output directory required")
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"news-importance-evaluate: {type(exc).__name__}: {exc}\n")
    print(json.dumps({
        "status": result["status"],
        "modelId": result["modelId"],
        "scoredArticles": result["scoredArticles"],
        "trainingDatasetEligible": False,
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
