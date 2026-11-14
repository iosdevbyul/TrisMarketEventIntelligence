"""Inspect unresolved review candidates against an existing research-only ledger."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from tris_market_event_intelligence.ticker_review_queue import (
    reconcile_ticker_review_queue,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconcile ticker candidates with human reviews (read-only)",
    )
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = reconcile_ticker_review_queue(args.proposals, args.ledger)
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error):
        parser.exit(2, "ticker-review-queue: reconciliation failed\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
