"""Single operational cycle, suitable for manually configured macOS launchd."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tris_market_event_intelligence.operations import OPERATIONAL_SOURCES, run_cycle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one bounded market-news collection cycle")
    default_base = Path.home() / "Library" / "Application Support" / "TrisMarketEventIntelligence"
    parser.add_argument("--db", type=Path, default=default_base / "news.sqlite3")
    parser.add_argument("--status", type=Path, default=default_base / "last-run.json")
    parser.add_argument("--sources", default=",".join(OPERATIONAL_SOURCES))
    parser.add_argument("--dry-run", action="store_true", help="Validate without network or DB")
    args = parser.parse_args()
    sources = tuple(name.strip() for name in args.sources.split(","))
    if args.dry_run:
        unknown = set(sources).difference(OPERATIONAL_SOURCES)
        if unknown or not sources:
            parser.error("Dry-run supports only known operational RSS sources")
        print(json.dumps({
            "mode": "dry_run", "db": str(args.db.expanduser()),
            "status_path": str(args.status.expanduser()),
            "sources": list(sources), "network_requests": 0,
            "background_service_installed": False,
        }, ensure_ascii=False))
        return
    report = run_cycle(
        args.db.expanduser(), status_path=args.status.expanduser(),
        selected_sources=sources,
    )
    print(json.dumps(report, ensure_ascii=False))
    if report["status"] in {"failed", "already_running"}:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
