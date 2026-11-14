"""Manual research review; no operational publication, no source DB writes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tris_market_event_intelligence.news_pilot_review_ledger import (
    NewsPilotReviewLedger,
)


def main() -> None:
    base = Path.home() / "Library/Application Support/TrisMarketEventIntelligence"
    parser = argparse.ArgumentParser(description="Private news-pilot editorial review")
    parser.add_argument("--source-db", type=Path, default=base / "news.sqlite3")
    parser.add_argument("--journal-db", type=Path, default=base / "news-pilot-review.sqlite3")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--list", action="store_true", help="Read-only evidence queue")
    actions.add_argument("--approve", type=int, metavar="EVIDENCE_ID")
    actions.add_argument("--reject", type=int, metavar="EVIDENCE_ID")
    parser.add_argument("--reviewer", help="Self-declared reviewer, not authenticated identity")
    parser.add_argument("--reason", help="Substantive reason for editorial decision")
    parser.add_argument("--commit", action="store_true", help="Explicitly persist review decision")
    args = parser.parse_args()
    if args.list and (args.commit or args.reviewer or args.reason):
        parser.error("--list does not accept --commit, --reviewer or --reason")
    if not args.list and (not args.reviewer or not args.reason):
        parser.error("Review requires --reviewer and --reason")
    ledger = NewsPilotReviewLedger(source_db=args.source_db, journal_db=args.journal_db)
    try:
        result = (
            ledger.list_entries() if args.list
            else ledger.decide(
                evidence_id=args.approve if args.approve is not None else args.reject,
                decision=(
                    "APPROVED_FOR_RESEARCH" if args.approve is not None else "REJECTED"
                ),
                reviewer=args.reviewer, reason=args.reason, commit=args.commit,
            )
        )
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"pilot-review: {type(exc).__name__}: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
