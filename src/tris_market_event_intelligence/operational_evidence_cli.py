"""One-command, offline and read-only operational evidence packet."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.operational_evidence import build_operational_evidence
from tris_market_event_intelligence.operations import OPERATIONAL_SOURCES
from tris_market_event_intelligence.session_operations import load_session_calendar


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only KR/US news collection, Gemma completion and freshness evidence",
    )
    base = Path.home() / "Library" / "Application Support" / "TrisMarketEventIntelligence"
    parser.add_argument("--db", type=Path, default=base / "news.sqlite3")
    parser.add_argument("--sources", default=",".join(OPERATIONAL_SOURCES))
    parser.add_argument("--sessions", type=Path)
    parser.add_argument("--markets", default="XKRX,XNYS,XNAS")
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument("--freshness-minutes", type=int, default=15)
    parser.add_argument("--source-success-minutes", type=int, default=15)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        now = datetime.fromisoformat(args.as_of) if args.as_of else datetime.now(UTC)
        result = build_operational_evidence(
            args.db.expanduser(), as_of=now,
            sources=tuple(x.strip() for x in args.sources.split(",")),
            markets=tuple(x.strip() for x in args.markets.split(",")),
            calendar=load_session_calendar(args.sessions.expanduser())
            if args.sessions else None,
            window_hours=args.window_hours,
            freshness_minutes=args.freshness_minutes,
            source_success_minutes=args.source_success_minutes,
        )
        if args.output is None:
            print(json.dumps(result, ensure_ascii=False))
        else:
            with args.output.open("x", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            print(json.dumps({
                "status": result["status"],
                "blocked_checks": result["blocked_checks"],
                "output": str(args.output),
            }, ensure_ascii=False))
        if result["status"] != "RESEARCH_OBSERVATIONS_ONLY":
            raise SystemExit(2)
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"operational-evidence: {type(exc).__name__}\n")


if __name__ == "__main__":
    main()
