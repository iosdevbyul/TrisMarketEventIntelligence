"""Explicit private research model export/replay; never auto-promotes a model."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from tris_market_event_intelligence.news_candidate_artifact import (
    build_candidate_artifact,
    compare_candidate_replay,
    write_candidate_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export or replay a journaled NEWS-only model candidate"
    )
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--export", action="store_true")
    operation.add_argument("--verify", action="store_true")
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--experiment-sha256")
    parser.add_argument("--artifact", type=Path, required=True)
    try:
        args = parser.parse_args()
        if args.export:
            if not args.experiment_sha256:
                parser.error("--export requires --experiment-sha256 from research journal")
            result = build_candidate_artifact(
                cohort_path=args.cohort, plan_path=args.holdout,
                review_path=args.review, journal_path=args.journal,
                experiment_sha256=args.experiment_sha256,
            )
            write_candidate_artifact(args.artifact, result)
            report = {
                "status": "PRIVATE_RESEARCH_ARTIFACT_CREATED",
                "experimentSha256": args.experiment_sha256,
                "payloadSha256": result["payloadSha256"],
                "productionEligible": False,
            }
        else:
            report = compare_candidate_replay(
                candidate_path=args.artifact, cohort_path=args.cohort,
                plan_path=args.holdout, review_path=args.review,
                journal_path=args.journal,
            )
    except (ValueError, TypeError, KeyError, OSError, sqlite3.Error,
            json.JSONDecodeError) as exc:
        parser.exit(2, f"news-candidate: {type(exc).__name__}: {exc}\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
