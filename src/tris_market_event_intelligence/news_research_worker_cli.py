"""Explicit opt-in, one-iteration CPU news research runner."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from tris_market_event_intelligence.news_research_worker import (
    audit_research_journal,
    research_news_once,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one bounded news importance research experiment"
    )
    parser.add_argument("--journal", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--audit", action="store_true")
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--holdout", type=Path)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    try:
        if args.audit:
            result = audit_research_journal(args.journal)
        else:
            if not all((args.cohort, args.holdout, args.review)):
                raise ValueError("One iteration requires cohort, holdout and human review")
            result = research_news_once(
                cohort_path=args.cohort, plan_path=args.holdout,
                review_path=args.review, journal_path=args.journal,
            )
    except (ValueError, TypeError, OSError, sqlite3.Error, KeyError,
            json.JSONDecodeError) as exc:
        parser.exit(2, f"news-research: {type(exc).__name__}: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
