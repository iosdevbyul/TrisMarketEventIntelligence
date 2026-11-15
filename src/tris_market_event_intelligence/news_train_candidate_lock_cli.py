"""Create-only NEWS TRAIN candidate lock and TEST budget audit."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from tris_market_event_intelligence.news_train_candidate_lock import (
    audit_candidate_lock,
    create_candidate_lock,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock one TRAIN-selected NEWS hypothesis")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--lock", action="store_true")
    action.add_argument("--audit", action="store_true")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--holdout", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--usage-ledger", type=Path)
    args = parser.parse_args()
    try:
        if args.lock:
            if not all((args.cohort, args.holdout, args.review, args.journal)):
                raise ValueError("Candidate lock requires all pinned inputs")
            if args.usage_ledger:
                raise ValueError("TEST budget is audited separately from selection")
            result = create_candidate_lock(
                output_path=args.manifest, cohort_path=args.cohort,
                plan_path=args.holdout, review_path=args.review,
                journal_path=args.journal,
            )
        else:
            result = audit_candidate_lock(
                lock_path=args.manifest, usage_ledger_path=args.usage_ledger,
                cohort_path=args.cohort, plan_path=args.holdout,
                review_path=args.review, journal_path=args.journal,
            )
    except (ValueError, TypeError, OSError, KeyError, sqlite3.Error,
            json.JSONDecodeError) as exc:
        parser.exit(2, f"news-candidate-lock: {type(exc).__name__}: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
