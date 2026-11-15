"""Opt-in isolated NEWS research subprocess with OS CPU and sampled RSS bounds."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from tris_market_event_intelligence.news_research_process_guard import (
    inspect_guard_status,
    run_guarded_news_research,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run/inspect a single safely guarded NEWS-only experiment process"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--status", action="store_true")
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--holdout", type=Path)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--lock-file", type=Path)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--max-cpu-seconds", type=int, default=30)
    parser.add_argument("--max-wall-seconds", type=int, default=300)
    parser.add_argument("--max-rss-mb", type=int, default=1024)
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--research-mode", choices=("legacy", "train-cv", "policy"),
                        default="legacy")
    args = parser.parse_args()
    try:
        if args.status:
            result = inspect_guard_status(args.status_file)
        else:
            if not all((args.cohort, args.holdout, args.review, args.journal,
                        args.lock_file)):
                raise ValueError("Run requires original news inputs, research journal and lock")
            result = run_guarded_news_research(
                cohort_path=args.cohort, plan_path=args.holdout,
                review_path=args.review, journal_path=args.journal,
                status_path=args.status_file, lock_path=args.lock_file,
                stop_file=args.stop_file,
                max_cpu_seconds=args.max_cpu_seconds,
                max_wall_seconds=args.max_wall_seconds,
                max_rss_mb=args.max_rss_mb,
                max_iterations=args.max_iterations,
                research_mode=args.research_mode,
            )
    except (ValueError, TypeError, OSError, KeyError,
            sqlite3.Error, json.JSONDecodeError) as exc:
        parser.exit(2, f"news-research-guard: {type(exc).__name__}: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
