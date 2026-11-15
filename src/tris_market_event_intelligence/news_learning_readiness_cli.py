"""Read-only local NEWS learning input and leakage readiness audit."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from tris_market_event_intelligence.news_learning_readiness import (
    inspect_news_learning_readiness,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect pinned NEWS learning readiness without exposing TEST"
    )
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--holdout", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--usage-ledger", type=Path)
    args = parser.parse_args()
    try:
        report = inspect_news_learning_readiness(
            cohort_path=args.cohort, plan_path=args.holdout,
            review_path=args.review, lock_path=args.manifest,
            journal_path=args.journal, usage_ledger_path=args.usage_ledger,
        )
    except (ValueError, TypeError, KeyError, OSError, sqlite3.Error,
            json.JSONDecodeError) as exc:
        parser.exit(2, f"news-learning-readiness: {type(exc).__name__}: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
