"""Read-only comparison of externally human-labelled semantic judgments."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from tris_market_event_intelligence.news_semantic_gold import evaluate_gold


def main() -> None:
    base = Path.home() / "Library/Application Support/TrisMarketEventIntelligence"
    parser = argparse.ArgumentParser(
        description="Evaluate source-pinned human judgments against semantic triage"
    )
    parser.add_argument("--source-db", type=Path, default=base / "news.sqlite3")
    parser.add_argument("--journal-db", type=Path, default=base / "news-pilot-review.sqlite3")
    parser.add_argument("--gold", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = evaluate_gold(
            source_db=args.source_db, journal_db=args.journal_db, gold_path=args.gold,
        )
    except (ValueError, OSError, sqlite3.Error, json.JSONDecodeError) as exc:
        parser.exit(2, f"news-semantic-gold: {type(exc).__name__}: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
