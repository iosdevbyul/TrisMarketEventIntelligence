"""Standalone, read-only integrated cycle watchdog command."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.integrated_watchdog import audit_integrated_schedule
from tris_market_event_intelligence.operations import OPERATIONAL_SOURCES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect last completed integrated run; no network, Ollama or launchctl",
    )
    base = Path.home() / "Library" / "Application Support" / "TrisMarketEventIntelligence"
    parser.add_argument("--status", type=Path, default=base / "integrated-last-run.json")
    parser.add_argument("--max-age-minutes", type=int, default=30)
    parser.add_argument("--sources", default=",".join(OPERATIONAL_SOURCES))
    parser.add_argument("--as-of", default=None)
    args = parser.parse_args()
    try:
        clock = datetime.fromisoformat(args.as_of) if args.as_of else datetime.now(UTC)
        result = audit_integrated_schedule(
            args.status.expanduser(), as_of=clock,
            expected_sources=tuple(s.strip() for s in args.sources.split(",")),
            max_age_seconds=args.max_age_minutes * 60,
        )
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"integrated-watchdog: {type(exc).__name__}\n")
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] != "OBSERVED_RECENT_SUCCESS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
