"""Read-only report of market-learning memory, evidence and rule status."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from tris_market_event_intelligence import learning_memory


def inspect(journal: Path) -> dict:
    if journal.is_symlink() or not journal.is_file():
        raise ValueError("Existing non-symlink learning journal required")
    uri = journal.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        db.execute("PRAGMA query_only=ON")
        required = {"memory_rules", "steps", "memory_rule_trials"}
        existing = {
            row[0] for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if not required.issubset(existing):
            raise ValueError("Journal has not yet initialized long-term learning memory")
        report = learning_memory.report(db)
        rules = []
        for row in db.execute(
            """SELECT r.id, r.version, r.family, r.sector, r.status, r.lesson,
                      r.failure_count, r.created_seq, r.eligible_after,
                      COUNT(t.seq), COALESCE(SUM(t.wins), 0),
                      COALESCE(SUM(t.losses), 0)
               FROM memory_rules r LEFT JOIN memory_rule_trials t
                    ON t.rule_id=r.id AND t.eligible=1 AND t.comparable=1
               GROUP BY r.id ORDER BY r.id"""
        ):
            rules.append(dict(zip(
                ("id", "version", "eventFamily", "sector", "status", "lesson",
                 "supportingFailures", "createdAtArticleIndex",
                 "eligibleAfter", "eligiblePairedCases", "wins", "losses"),
                row, strict=True,
            )))
        groups = [
            {"eventFamily": family, "sector": sector,
             "cases": count, "failedPredictions": wrong}
            for family, sector, count, wrong in db.execute(
                """SELECT family, sector, COUNT(*), SUM(wrong_count)
                   FROM memory_failure_cases GROUP BY family, sector
                   ORDER BY COUNT(*) DESC, family, sector"""
            )
        ]
        return {
            "status": "REPORT_ONLY",
            "learningMemory": report,
            "failurePatterns": groups,
            "rules": rules,
            "notes": [
                "Rule ACTIVE status requires later paired PIT-eligible evidence, "
                "not simply having been suggested by the model.",
                "The retrospective feedback stream can use later observed outcomes "
                "in earlier article predictions. It is not a leakage-free backtest.",
                "Holdout evaluation has not been executed by this command.",
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect sequential learning memory")
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    try:
        result = inspect(args.journal)
        if args.output is not None:
            if args.output.is_symlink() or not args.output.parent.is_dir():
                raise ValueError("Regular existing output directory required")
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, sqlite3.Error) as error:
        parser.exit(2, f"learning-memory-report: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    main()
