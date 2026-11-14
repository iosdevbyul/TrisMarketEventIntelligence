"""Explicit one-shot import of a real pilot report into an isolated review journal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tris_market_event_intelligence.news_pilot_evidence_journal import (
    NewsPilotEvidenceJournal,
)


def main() -> None:
    base = Path.home() / "Library/Application Support/TrisMarketEventIntelligence"
    parser = argparse.ArgumentParser(
        description="Validate and optionally archive source-linked pilot evidence"
    )
    parser.add_argument("--source-db", type=Path, default=base / "news.sqlite3")
    parser.add_argument("--journal-db", type=Path, default=base / "news-pilot-review.sqlite3")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--commit", action="store_true",
        help="Explicitly create/update separate private review journal",
    )
    args = parser.parse_args()
    try:
        journal = NewsPilotEvidenceJournal(
            source_db=args.source_db, journal_db=args.journal_db,
        )
        result = journal.import_report(args.report, commit=args.commit)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"pilot-evidence: {type(exc).__name__}: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
