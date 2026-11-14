"""Local read-only news timing audit; no feed fetch or DB migration."""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.news_latency import audit_news_latency


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit collected news source latency")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--as-of", default=None, help="Optional ISO timezone-aware audit timestamp")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    try:
        when = datetime.fromisoformat(args.as_of) if args.as_of else datetime.now(UTC)
        report = audit_news_latency(args.db, as_of=when)
        if args.output is None:
            print(json.dumps(report, ensure_ascii=False))
        else:
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            print(json.dumps({
                "status": "audit_only", "report": str(args.output),
                "sources": report["source_count"],
            }))
    except (OSError, ValueError, TypeError, sqlite3.Error) as exc:
        parser.exit(2, f"news-latency-audit: {exc}\n")


if __name__ == "__main__":
    main()
