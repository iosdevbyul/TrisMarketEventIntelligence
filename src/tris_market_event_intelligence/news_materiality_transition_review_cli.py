"""Prepare source-blind pair-review template or evaluate validation pairs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tris_market_event_intelligence.news_materiality_transition_review import (
    evaluate_transition_review,
    prepare_transition_review,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Human-only paired NEWS materiality comparison, no TEST gold"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--evaluate", action="store_true")
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.prepare:
            if args.review is not None:
                raise ValueError("Review path is only permitted for evaluation")
            result = prepare_transition_review(
                cohort_path=args.cohort, holdout_path=args.holdout,
                report_path=args.report,
            )
        else:
            if args.review is None:
                raise ValueError("Completed human review required for evaluation")
            result = evaluate_transition_review(
                cohort_path=args.cohort, holdout_path=args.holdout,
                report_path=args.report, human_review_path=args.review,
            )
        if args.output.exists() or args.output.is_symlink() or (
            not args.output.parent.is_dir()
        ):
            raise ValueError("New private output file in existing directory required")
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(args.output, flags, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
        except BaseException:
            args.output.unlink(missing_ok=True)
            raise
    except (ValueError, TypeError, KeyError, OSError,
            json.JSONDecodeError) as exc:
        parser.exit(2, f"news-materiality-transition: {type(exc).__name__}: {exc}\n")
    print(json.dumps({
        "status": result.get("status", "HUMAN_REVIEW_TEMPLATE_UNFILLED"),
        "selectedPairs": len(result["judgments"]) if args.prepare else (
            result["selectedPairsWithinValidation"]
        ),
        "modelImprovementValidated": False,
        "promotionAllowed": False,
        "stockPricesUsed": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
