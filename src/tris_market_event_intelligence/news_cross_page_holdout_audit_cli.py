"""Offline, create-only audit over complete news cohort page sequence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tris_market_event_intelligence.news_cross_page_holdout_audit import (
    audit_cross_page_news,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit cross-page news-only holdout duplicates")
    parser.add_argument("--cohort", type=Path, action="append", required=True)
    parser.add_argument("--holdout", type=Path, action="append", required=True)
    parser.add_argument("--threshold", type=float, default=0.78)
    parser.add_argument("--max-days", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.cohort) != len(args.holdout):
        parser.error("Exactly one --holdout required for each --cohort, in order")
    try:
        result = audit_cross_page_news(
            pages=list(zip(args.cohort, args.holdout, strict=True)),
            threshold=args.threshold, max_days=args.max_days,
        )
        if args.output.is_symlink() or not args.output.parent.is_dir():
            raise ValueError("Existing ordinary output parent required")
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"news-cross-page: {type(exc).__name__}: {exc}\n")
    print(json.dumps({
        "status": result["status"], "pageCount": result["pageCount"],
        "crossPageCandidates": result["crossPageCandidates"],
        "trainingDatasetEligible": False, "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
