"""No-network operator preflight: reviewed exchange sessions plus local source health."""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.operations import OPERATIONAL_SOURCES
from tris_market_event_intelligence.session_operations import (
    load_session_calendar,
    operational_readiness,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only KR/US session and source readiness (not live-news proof)",
    )
    base = Path.home() / "Library" / "Application Support" / "TrisMarketEventIntelligence"
    parser.add_argument("--db", type=Path, default=base / "news.sqlite3")
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--markets", default="XKRX,XNYS,XNAS")
    parser.add_argument("--sources", default=",".join(OPERATIONAL_SOURCES))
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--max-success-minutes", type=int, default=15)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    try:
        clock = datetime.fromisoformat(args.as_of) if args.as_of else datetime.now(UTC)
        report = operational_readiness(
            args.db.expanduser(),
            calendar=load_session_calendar(args.sessions.expanduser()),
            now=clock,
            markets=tuple(s.strip() for s in args.markets.split(",")),
            sources=tuple(s.strip() for s in args.sources.split(",")),
            max_success_age_seconds=args.max_success_minutes * 60,
        )
        if args.output is None:
            print(json.dumps(report, ensure_ascii=False))
        else:
            with args.output.open("x", encoding="utf-8") as out:
                json.dump(report, out, ensure_ascii=False, indent=2)
                out.write("\n")
            print(json.dumps({
                "status": report["decision"], "output": str(args.output),
            }))
        if report["decision"] == "BLOCKED":
            raise SystemExit(2)
    except (OSError, ValueError, TypeError, sqlite3.Error) as exc:
        parser.exit(2, f"market-ops-readiness: {type(exc).__name__}\n")


if __name__ == "__main__":
    main()
