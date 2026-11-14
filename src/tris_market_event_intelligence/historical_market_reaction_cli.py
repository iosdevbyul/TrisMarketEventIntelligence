"""Explicit offline reader for historical article/market response comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tris_market_event_intelligence.historical_market_reaction_join import (
    join_historical_market_reactions,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare historical all-news inventory to supplied minute-bar reactions"
    )
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--links", type=Path, required=True)
    parser.add_argument("--minute-manifest", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--halts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = join_historical_market_reactions(
            cohort_path=args.cohort, links_path=args.links,
            minute_manifest=args.minute_manifest, sessions_path=args.sessions,
            halts_path=args.halts,
        )
        if args.output.is_symlink() or not args.output.parent.is_dir():
            raise ValueError("Existing ordinary output directory required")
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    except (ValueError, TypeError, OSError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, f"historical-market-reactions: {type(exc).__name__}: {exc}\n")
    print(json.dumps({
        "status": result["status"],
        "articlesInCohortPage": result["articlesInCohortPage"],
        "articlesWithPointInTimeLinks": result["articlesWithPointInTimeLinks"],
        "labelCounts": result["labelCounts"],
        "trainingDatasetEligible": False,
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
