"""Explicit research-only export of the archived all-news cohort."""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from tris_market_event_intelligence.historical_news_cohort import historical_news_cohort


def main() -> None:
    base = Path.home() / "Library/Application Support/TrisMarketEventIntelligence"
    parser = argparse.ArgumentParser(
        description="Export read-only historical news, including non-event articles"
    )
    parser.add_argument("--source-db", type=Path, default=base / "news.sqlite3")
    parser.add_argument("--as-of", required=True, help="Timezone-aware historical cutoff")
    parser.add_argument("--after-id", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument(
        "--output", type=Path, required=True,
        help="New local JSON file, never overwrites an existing report",
    )
    args = parser.parse_args()
    try:
        as_of = datetime.fromisoformat(args.as_of)
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("Explicit timezone required")
        if not args.output.parent.is_dir() or args.output.is_symlink():
            raise ValueError("Existing ordinary output parent required")
        result = historical_news_cohort(
            source_db=args.source_db, as_of=as_of, after_id=args.after_id, limit=args.limit,
        )
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(2, f"historical-news-cohort: {type(exc).__name__}: {exc}\n")
    print(json.dumps({
        "status": result["status"],
        "asOf": result["asOf"],
        "totalObservedByCutoff": result["totalObservedByCutoff"],
        "pageCount": len(result["items"]),
        "nextCursor": result["nextCursor"],
        "hasMore": result["hasMore"],
        "trainingDatasetEligible": False,
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
