"""Read-only issuer matching proposals; never persist or train on links."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from tris_market_event_intelligence.event_ticker_proposals import (
    load_reviewed_issuers,
    propose_event_tickers,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview reviewed-listing-based event→stock candidate links offline",
    )
    base = Path.home() / "Library" / "Application Support" / "TrisMarketEventIntelligence"
    parser.add_argument("--db", type=Path, default=base / "news.sqlite3")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    try:
        issuers = load_reviewed_issuers(args.registry.expanduser())
        result = propose_event_tickers(
            args.db.expanduser(), issuers, limit=args.limit,
        )
    except (ValueError, TypeError, OSError, sqlite3.Error, UnicodeError) as exc:
        parser.exit(2, f"event-ticker-proposals: {type(exc).__name__}\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
