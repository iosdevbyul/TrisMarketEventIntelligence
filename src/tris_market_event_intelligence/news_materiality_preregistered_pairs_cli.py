"""Pre-register blind NEWS pairs before local-model comparisons."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tris_market_event_intelligence.news_materiality_preregistered_pairs import (
    evaluate_preregistered_pairs,
    prepare_preregistered_pairs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed, model-independent news-pair sampling and human audit"
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--evaluate", action="store_true")
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--holdout", required=True, type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.prepare:
            if args.review is not None or args.report is not None:
                raise ValueError("Pre-registration must not access models or gold")
            result = prepare_preregistered_pairs(
                cohort_path=args.cohort, holdout_path=args.holdout,
            )
        else:
            if args.review is None or args.report is None:
                raise ValueError("Human review and independent model report required")
            result = evaluate_preregistered_pairs(
                cohort_path=args.cohort, holdout_path=args.holdout,
                review_path=args.review, report_path=args.report,
            )
        if args.output.is_symlink() or args.output.exists() or (
            not args.output.parent.is_dir()
        ):
            raise ValueError("Private output must be a new file in existing directory")
        flags = os.O_EXCL | os.O_WRONLY | os.O_CREAT
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
    except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"news-preregistered-pairs: {type(exc).__name__}: {exc}\n")
    print(json.dumps({
        "status": result.get("status", "BLIND_PAIRS_NOT_HUMAN_REVIEWED"),
        "selectedPairs": (len(result["judgments"]) if args.prepare
                          else result["selectedValidationPairs"]),
        "modelPerformanceImprovementValidated": False,
        "testGoldScored": False, "promotionAllowed": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
