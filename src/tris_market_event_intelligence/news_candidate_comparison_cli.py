"""Compare private NEWS-only research candidates, without a promote action."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from tris_market_event_intelligence.news_candidate_comparison import (
    compare_news_candidates,
    write_comparison_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank replay-verified NEWS research models for human review"
    )
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = compare_news_candidates(
            candidate_paths=args.candidate, cohort_path=args.cohort,
            plan_path=args.holdout, review_path=args.review,
            journal_path=args.journal,
        )
        write_comparison_report(args.output, report)
    except (ValueError, TypeError, KeyError, OSError, sqlite3.Error,
            json.JSONDecodeError) as exc:
        parser.exit(2, f"news-candidate-comparison: {type(exc).__name__}: {exc}\n")
    print(json.dumps({
        "status": report["status"],
        "candidateCount": report["candidateCount"],
        "validationArticles": report["validationArticles"],
        "promotionAllowed": False,
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
