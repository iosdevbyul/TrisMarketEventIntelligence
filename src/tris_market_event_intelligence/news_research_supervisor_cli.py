"""Foreground NEWS-only research supervisor CLI; never installs launchd."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from tris_market_event_intelligence.news_research_supervisor import (
    read_supervisor_status,
    run_supervised_news_research,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect or run a resource-sampled, finite NEWS-only research supervisor"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true")
    group.add_argument("--status", action="store_true")
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--holdout", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--max-cpu-seconds", type=float, default=600)
    parser.add_argument("--max-peak-rss-mb", type=float, default=1024)
    parser.add_argument("--stale-after-seconds", type=int, default=600)
    args = parser.parse_args()
    try:
        if args.status:
            result = read_supervisor_status(
                args.status_file, stale_after_seconds=args.stale_after_seconds,
            )
        else:
            if not all((
                args.cohort, args.holdout, args.review, args.journal, args.lock_file,
            )):
                raise ValueError("Run requires cohort, holdout, review, journal and lock file")
            result = run_supervised_news_research(
                cohort_path=args.cohort, plan_path=args.holdout,
                review_path=args.review, journal_path=args.journal,
                status_path=args.status_file, lock_path=args.lock_file,
                stop_file=args.stop_file, max_iterations=args.max_iterations,
                max_seconds=args.max_seconds, max_cpu_seconds=args.max_cpu_seconds,
                max_peak_rss_mb=args.max_peak_rss_mb,
            )
    except (ValueError, TypeError, OSError, KeyError, sqlite3.Error,
            json.JSONDecodeError) as exc:
        parser.exit(2, f"news-research-supervisor: {type(exc).__name__}: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
