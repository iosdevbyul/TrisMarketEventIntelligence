"""Manual, opt-in exchange directory capture + ticker candidate workflow."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import httpx

from tris_market_event_intelligence.listing_research_workflow import (
    acquire_and_reconcile_listings,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquire actual listings into a private, create-only review bundle",
    )
    parser.add_argument("--fetch-us", action="store_true",
                        help="Explicit HTTPS download from two official NasdaqTrader URLs")
    parser.add_argument("--krx-reviewed", type=Path,
                        help="Operator-normalized KRX common stock CSV, never downloaded")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="New, nonexistent private directory under an existing folder")
    parser.add_argument("--news-db", type=Path,
                        help="Existing news SQLite, read only; absent means snapshot-only")
    parser.add_argument("--review-ledger", type=Path,
                        help="Existing reviewed ledger; requires --news-db")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    try:
        result = acquire_and_reconcile_listings(
            args.output_dir.expanduser(), fetch_us=args.fetch_us,
            krx_reviewed=args.krx_reviewed.expanduser() if args.krx_reviewed else None,
            news_db=args.news_db.expanduser() if args.news_db else None,
            ledger=args.review_ledger.expanduser() if args.review_ledger else None,
            limit=args.limit,
        )
    except ValueError as exc:
        # Show only bounded validation messages, never raw source records.
        reason = str(exc).replace("\n", " ").replace("\r", " ")[:180]
        parser.exit(2, f"listing-research: {reason}\n")
    except (OSError, KeyError, UnicodeError, TypeError,
            httpx.HTTPError, sqlite3.Error, json.JSONDecodeError) as exc:
        parser.exit(2, f"listing-research: {type(exc).__name__}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
