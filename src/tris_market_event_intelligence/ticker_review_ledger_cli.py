"""Explicit research-only review ledger import and read-only inspection."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from tris_market_event_intelligence.ticker_review_ledger import (
    import_review_decisions,
    list_review_decisions,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Store or inspect post-hoc review decisions in an isolated SQLite DB",
    )
    parser.add_argument("--ledger", type=Path, required=True)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--import-reviewed", action="store_true")
    actions.add_argument("--list", action="store_true")
    parser.add_argument("--proposals", type=Path)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--event-id", type=int)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    if args.import_reviewed and (
        args.proposals is None or args.decisions is None or args.event_id is not None
    ):
        parser.error("Import requires both --proposals and --decisions, not --event-id")
    if args.list and (args.proposals is not None or args.decisions is not None):
        parser.error("Read-only listing cannot include review source files")
    try:
        if args.import_reviewed:
            report = import_review_decisions(
                args.ledger, args.proposals, args.decisions,
            )
        else:
            report = list_review_decisions(
                args.ledger, event_id=args.event_id, limit=args.limit,
            )
    except (OSError, ValueError, TypeError, sqlite3.Error, KeyError) as exc:
        parser.exit(2, f"ticker-review-ledger: {type(exc).__name__}\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
