"""Manual finite news-only TRAIN-fold exploration, no VALIDATION tuning."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from tris_market_event_intelligence.news_train_cv_exploration import (
    audit_train_exploration,
    explore_news_train_once,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Research news importance features with TRAIN-only cross-validation"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--once", action="store_true")
    action.add_argument("--batch", type=int, metavar="ITERATIONS")
    action.add_argument("--audit", action="store_true")
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--holdout", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--journal", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.audit:
            result = audit_train_exploration(args.journal)
        else:
            if not all((args.cohort, args.holdout, args.review)):
                raise ValueError("Training-only research requires cohort, holdout and review")
            count = 1 if args.once else args.batch
            if type(count) is not int or not 1 <= count <= 18:
                raise ValueError("One batch must run between 1 and 18 finite experiments")
            results = []
            for _ in range(count):
                result = explore_news_train_once(
                    cohort_path=args.cohort, plan_path=args.holdout,
                    review_path=args.review, journal_path=args.journal,
                )
                results.append(result)
                if result["status"] != "TRAIN_ONLY_CV_EXPLORATORY":
                    break
            result = {
                "schemaVersion": "news_train_cv_batch_v1",
                "status": results[-1]["status"],
                "newResearchExperiments": sum(
                    int(row.get("newResearchExperiment", False)) for row in results
                ),
                "attemptedIterations": len(results),
                "lastIteration": results[-1],
                "modelImprovementValidated": False,
                "validationGoldScored": False,
                "testGoldScored": False,
                "modelPromotionAttempted": False,
                "stockPricesUsed": False,
            }
    except (ValueError, TypeError, OSError, sqlite3.Error, KeyError,
            json.JSONDecodeError) as exc:
        parser.exit(2, f"news-train-cv: {type(exc).__name__}: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
