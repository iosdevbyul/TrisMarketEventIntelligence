"""Analyze actual Mac captured listing snapshots against existing news."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from tris_market_event_intelligence.listing_news_validation import (
    validate_existing_listing_news,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline source-verified listing + news diagnostics and review pages",
    )
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--news-db", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--review-ledger", type=Path)
    parser.add_argument("--max-pages", type=int, default=16)
    args = parser.parse_args()
    try:
        result = validate_existing_listing_news(
            args.bundle.expanduser(), args.news_db.expanduser(),
            args.output_dir.expanduser(),
            ledger=args.review_ledger.expanduser() if args.review_ledger else None,
            max_pages=args.max_pages,
        )
    except ValueError as exc:
        reason = str(exc).replace("\n", " ").replace("\r", " ")[:180]
        parser.exit(2, f"listing-validation: {reason}\n")
    except (OSError, TypeError, KeyError, UnicodeError, sqlite3.Error) as exc:
        parser.exit(2, f"listing-validation: {type(exc).__name__}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
