"""Manual finite NEWS-only hypotheses on protected TRAIN folds."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from tris_market_event_intelligence.news_research_policy import (
    MAX_HYPOTHESES,
    audit_policy,
    run_policy_once,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finite research-only NEWS hypotheses")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--batch", type=int, metavar="COUNT")
    actions.add_argument("--audit", action="store_true")
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--holdout", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--journal", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.audit:
            report = audit_policy(args.journal)
        else:
            if type(args.batch) is not int or not 1 <= args.batch <= MAX_HYPOTHESES:
                raise ValueError("Invalid finite TRAIN hypothesis batch size")
            if not all((args.cohort, args.holdout, args.review)):
                raise ValueError("Pinned private NEWS inputs required")
            results = []
            for _ in range(args.batch):
                item = run_policy_once(
                    cohort_path=args.cohort, plan_path=args.holdout,
                    review_path=args.review, journal_path=args.journal,
                )
                results.append(item)
                if not item["newResearchExperiment"]:
                    break
            report = {
                "schemaVersion": "news_research_policy_batch_v1",
                "status": results[-1]["status"],
                "newResearchExperiments": sum(
                    int(r["newResearchExperiment"]) for r in results
                ),
                "attemptedIterations": len(results),
                "lastIteration": results[-1],
                "promotionAllowed": False,
                "independentImprovementVerified": False,
                "stockPricesUsed": False,
            }
    except (ValueError, TypeError, OSError, KeyError, sqlite3.Error,
            json.JSONDecodeError) as exc:
        parser.exit(2, f"news-research-policy: {type(exc).__name__}: {exc}\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
