"""Read-only multi-source freshness and outage report."""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.operations import OPERATIONAL_SOURCES
from tris_market_event_intelligence.source_health import audit_source_health


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check verified local source polling telemetry without fetching news",
    )
    base = Path.home() / "Library" / "Application Support" / "TrisMarketEventIntelligence"
    parser.add_argument("--db", type=Path, default=base / "news.sqlite3")
    parser.add_argument("--sources", default=",".join(OPERATIONAL_SOURCES))
    parser.add_argument("--max-success-minutes", type=int, default=15)
    parser.add_argument("--max-failures", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    sources = tuple(source.strip() for source in args.sources.split(","))
    if not all(sources) or len(sources) != len(set(sources)):
        parser.error("Use unique non-empty source IDs")
    try:
        report = audit_source_health(
            args.db.expanduser(), expected_sources=sources,
            now=datetime.now(UTC),
            max_success_age_seconds=args.max_success_minutes * 60,
            maximum_consecutive_failures=args.max_failures,
        )
        if args.output is None:
            print(json.dumps(report, ensure_ascii=False))
        else:
            with args.output.open("x", encoding="utf-8") as out:
                json.dump(report, out, ensure_ascii=False, indent=2)
                out.write("\n")
            print(json.dumps({
                "status": report["status"], "report": str(args.output),
                "source_count": len(report["sources"]),
            }))
        if report["status"] != "OK":
            raise SystemExit(2)
    except (OSError, ValueError, TypeError, sqlite3.Error) as error:
        parser.exit(2, f"source-health: {type(error).__name__}\n")


if __name__ == "__main__":
    main()
