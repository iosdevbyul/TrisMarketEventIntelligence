"""Offline candidate audit of near-duplicate news across holdout splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tris_market_event_intelligence.news_near_duplicate_audit import (
    audit_news_near_duplicates,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find human-review-only news title overlaps across holdout splits"
    )
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.78)
    parser.add_argument("--max-days", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit_news_near_duplicates(
            cohort_path=args.cohort, holdout_path=args.holdout,
            threshold=args.threshold, max_days=args.max_days,
        )
        if args.output.is_symlink() or not args.output.parent.is_dir():
            raise ValueError("Existing ordinary output directory required")
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"news-near-duplicates: {type(exc).__name__}: {exc}\n")
    print(json.dumps({
        "status": result["status"],
        "crossPartitionCandidates": result["crossPartitionCandidates"],
        "trainingDatasetEligible": False,
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
