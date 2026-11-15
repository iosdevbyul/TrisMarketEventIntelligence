"""CLI for bounded news-only research loops and cooperative stop flags."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from tris_market_event_intelligence.news_research_loop import (
    request_research_stop,
    run_news_research_loop,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a bounded, recoverable NEWS-only research loop"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true")
    group.add_argument("--stop", type=Path, metavar="STOP_FILE")
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--holdout", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--max-backoff-seconds", type=float, default=60.0)
    args = parser.parse_args()
    try:
        if args.stop is not None:
            result = request_research_stop(args.stop)
        else:
            if not all((args.cohort, args.holdout, args.review, args.journal)):
                raise ValueError("Research loop requires news cohort, holdout, review, journal")
            result = run_news_research_loop(
                cohort_path=args.cohort, plan_path=args.holdout,
                review_path=args.review, journal_path=args.journal,
                max_iterations=args.max_iterations, max_seconds=args.max_seconds,
                interval_seconds=args.interval_seconds,
                max_backoff_seconds=args.max_backoff_seconds,
                stop_file=args.stop_file,
            )
    except (ValueError, TypeError, OSError, KeyError, sqlite3.Error,
            json.JSONDecodeError) as exc:
        parser.exit(2, f"news-research-loop: {type(exc).__name__}: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
