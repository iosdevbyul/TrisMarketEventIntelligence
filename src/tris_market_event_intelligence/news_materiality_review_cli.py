"""Prepare human magnitude reviews or score pinned descriptive VALIDATION."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from tris_market_event_intelligence.news_materiality_review import (
    evaluate_materiality_validation,
    prepare_materiality_review,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Human-reviewed NEWS magnitude: direction-independent and price-free"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--holdout", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.prepare:
            if any((args.holdout, args.review, args.predictions)):
                raise ValueError("Review preparation takes only a pinned news cohort")
            result = prepare_materiality_review(args.cohort)
        else:
            if not all((args.holdout, args.review, args.predictions)):
                raise ValueError("Validation scoring requires plan, human review, and predictions")
            result = evaluate_materiality_validation(
                cohort_path=args.cohort, plan_path=args.holdout,
                reviews_path=args.review, predictions_path=args.predictions,
            )
        if args.output.is_symlink() or args.output.exists() or (
            not args.output.parent.is_dir()
        ):
            raise ValueError("Private output must be a new file in an existing directory")
        import os

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(args.output, flags, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
        except BaseException:
            args.output.unlink(missing_ok=True)
            raise
    except (ValueError, TypeError, KeyError, OSError, ValidationError,
            json.JSONDecodeError) as exc:
        parser.exit(2, f"news-materiality: {type(exc).__name__}: {exc}\n")
    print(json.dumps({
        "status": result.get("status", "BLANK_HUMAN_REVIEW_TEMPLATE"),
        "output": str(args.output), "promotionAllowed": False,
        "stockPricesUsed": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
