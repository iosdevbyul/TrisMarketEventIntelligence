"""Opt-in human checklist notes for the separate news research journal."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from tris_market_event_intelligence.news_pilot_checklist_ledger import (
    STATES,
    NewsPilotChecklistLedger,
)


def main() -> None:
    base = Path.home() / "Library/Application Support/TrisMarketEventIntelligence"
    parser = argparse.ArgumentParser(
        description="Record source-documentation notes; no publication or fact certification"
    )
    parser.add_argument("--source-db", type=Path, default=base / "news.sqlite3")
    parser.add_argument("--journal-db", type=Path, default=base / "news-pilot-review.sqlite3")
    parser.add_argument("--evidence-id", type=int, required=True)
    parser.add_argument("--check", required=True)
    parser.add_argument("--status", choices=sorted(STATES), required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--note", required=True)
    parser.add_argument("--source-url")
    parser.add_argument("--source-excerpt")
    parser.add_argument("--commit", action="store_true")
    args = parser.parse_args()
    try:
        result = NewsPilotChecklistLedger(
            source_db=args.source_db, journal_db=args.journal_db,
        ).record(
            evidence_id=args.evidence_id, check_code=args.check,
            review_status=args.status, reviewer=args.reviewer, note=args.note,
            source_url=args.source_url, source_excerpt=args.source_excerpt,
            commit=args.commit,
        )
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(2, f"pilot-checklist: {type(exc).__name__}: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
