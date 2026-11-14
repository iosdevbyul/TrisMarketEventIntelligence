"""CLI for isolated, append-only news research holdout reservations."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from tris_market_event_intelligence.news_test_usage_ledger import (
    audit_news_test_usage,
    initialize_usage_ledger,
    reserve_news_test_use,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate final news-only TEST exposure through a private ledger"
    )
    parser.add_argument("--ledger", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--init", action="store_true")
    group.add_argument("--audit", action="store_true")
    group.add_argument("--reserve", choices=("TRAIN", "VALIDATION", "TEST_FINAL"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--experiment-id")
    parser.add_argument("--candidate-sha256")
    parser.add_argument("--reason")
    args = parser.parse_args()
    try:
        if args.init:
            result = initialize_usage_ledger(args.ledger)
        elif args.audit:
            result = audit_news_test_usage(args.ledger)
        else:
            if not all((
                args.plan, args.experiment_id, args.candidate_sha256, args.reason,
            )):
                raise ValueError("Reservation requires plan, experiment ID, SHA, and reason")
            result = reserve_news_test_use(
                ledger_path=args.ledger, plan_path=args.plan,
                experiment_id=args.experiment_id,
                candidate_sha256=args.candidate_sha256,
                purpose=args.reserve, justification=args.reason,
            )
    except (ValueError, TypeError, KeyError, OSError, sqlite3.Error,
            json.JSONDecodeError) as exc:
        parser.exit(2, f"news-test-usage: {type(exc).__name__}: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
