"""Explicit read-only view of a fully audited AI-news research evidence packet."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from tris_market_event_intelligence.news_pilot_review_packet import NewsPilotReviewPacket


def main() -> None:
    base = Path.home() / "Library/Application Support/TrisMarketEventIntelligence"
    parser = argparse.ArgumentParser(
        description="Inspect original news, hypotheses and independent check gaps"
    )
    parser.add_argument("--source-db", type=Path, default=base / "news.sqlite3")
    parser.add_argument("--journal-db", type=Path, default=base / "news-pilot-review.sqlite3")
    parser.add_argument("--evidence-id", type=int, required=True)
    args = parser.parse_args()
    try:
        report = NewsPilotReviewPacket(
            source_db=args.source_db, journal_db=args.journal_db,
        ).build(evidence_id=args.evidence_id)
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(2, f"pilot-review-packet: {type(exc).__name__}: {exc}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
