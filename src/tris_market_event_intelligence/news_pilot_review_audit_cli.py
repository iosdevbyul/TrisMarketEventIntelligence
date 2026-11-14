"""Inspect and audit the separate research review journal; read-only."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from tris_market_event_intelligence.news_pilot_review_audit import (
    NewsPilotReviewAuditor,
    STATES,
)


def main() -> None:
    base = Path.home() / "Library/Application Support/TrisMarketEventIntelligence"
    parser = argparse.ArgumentParser(
        description="Read-only source-linked research-review audit"
    )
    parser.add_argument("--source-db", type=Path, default=base / "news.sqlite3")
    parser.add_argument("--journal-db", type=Path, default=base / "news-pilot-review.sqlite3")
    parser.add_argument(
        "--status", choices=sorted(STATES),
        help="Filter by pending, approved-for-research, or rejected",
    )
    parser.add_argument("--after-id", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    try:
        result = NewsPilotReviewAuditor(
            source_db=args.source_db, journal_db=args.journal_db,
        ).audit(status=args.status, after_id=args.after_id, limit=args.limit)
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(2, f"pilot-review-audit: {type(exc).__name__}: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
