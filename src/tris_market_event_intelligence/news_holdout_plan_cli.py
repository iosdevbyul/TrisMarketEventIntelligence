"""Read-only, create-only news holdout planning CLI."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from tris_market_event_intelligence.news_holdout_plan import plan_news_holdout


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan chronological news-only training/validation/test partitions"
    )
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--review", type=Path, default=None,
                        help="Optional private human-label review, used for TRAIN cutoff audit")
    parser.add_argument("--train-end", type=datetime.fromisoformat, required=True)
    parser.add_argument("--validation-end", type=datetime.fromisoformat, required=True)
    parser.add_argument("--embargo-hours", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = plan_news_holdout(
            args.cohort, train_end=args.train_end,
            validation_end=args.validation_end,
            embargo_hours=args.embargo_hours, review_path=args.review,
        )
        if args.output.is_symlink() or not args.output.parent.is_dir():
            raise ValueError("Existing ordinary output directory required")
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except (ValueError, TypeError, OSError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, f"news-holdout: {type(exc).__name__}: {exc}\n")
    print(json.dumps({
        "status": result["status"],
        "counts": result["counts"],
        "groupsConsidered": result["groupsConsidered"],
        "quarantinedArticles": result["quarantinedArticles"],
        "trainingDatasetEligible": False,
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
