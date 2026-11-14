"""Create-only CLI for conservative NEWS-only chronological research partitions."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from tris_market_event_intelligence.news_chronological_partitions import (
    split_news_chronologically,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split all archived news by time and quarantine cross-window stories"
    )
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--train-end", type=datetime.fromisoformat, required=True)
    parser.add_argument("--validation-end", type=datetime.fromisoformat, required=True)
    parser.add_argument("--embargo-hours", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = split_news_chronologically(
            cohort_path=args.cohort, review_path=args.review,
            train_end=args.train_end, validation_end=args.validation_end,
            embargo_hours=args.embargo_hours,
        )
        if args.output.is_symlink() or not args.output.parent.is_dir():
            raise ValueError("Existing ordinary output directory required")
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"news-partitions: {type(exc).__name__}: {exc}\n")
    print(json.dumps({
        "status": result["status"],
        "partitionCounts": result["partitionCounts"],
        "groupCount": result["groupCount"],
        "trainingDatasetEligible": False,
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
