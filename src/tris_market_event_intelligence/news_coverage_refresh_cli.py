"""One-shot private NEWS coverage refresh; no human labels or model run."""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.news_coverage_refresh import (
    refresh_news_coverage,
)


def main() -> None:
    base = Path.home() / "Library/Application Support/TrisMarketEventIntelligence"
    parser = argparse.ArgumentParser(
        description="Verify NEWS growth and persist a new cohort only if new articles exist"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--source-db", type=Path, default=base / "news.sqlite3")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--as-of", type=str, default=None)
    args = parser.parse_args()
    try:
        cutoff = datetime.fromisoformat(args.as_of) if args.as_of else datetime.now(UTC)
        result = refresh_news_coverage(
            baseline_path=args.baseline,
            source_db=args.source_db,
            output_path=args.output,
            as_of=cutoff,
        )
    except (ValueError, TypeError, KeyError, OSError, sqlite3.Error,
            json.JSONDecodeError) as exc:
        parser.exit(2, f"news-coverage-refresh: {type(exc).__name__}: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
