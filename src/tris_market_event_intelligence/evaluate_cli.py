"""Run the fixed November 13 headline replay and optional SQLite collection audit."""

import argparse
import json
import os
from pathlib import Path

from tris_market_event_intelligence.evaluation import replay, stored_coverage


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate verified-source news headline coverage")
    parser.add_argument(
        "--benchmark", type=Path,
        default=Path(__file__).resolve().parents[2] / "benchmarks"
        / "2026-11-13-headlines.json",
    )
    parser.add_argument(
        "--db", type=Path,
        default=Path(os.getenv("MARKET_EVENT_DB", "data/news.sqlite3")),
    )
    parser.add_argument("--audit-db", action="store_true")
    args = parser.parse_args()
    result = replay(args.benchmark)
    if args.audit_db:
        try:
            result["collection_audit"] = stored_coverage(args.benchmark, args.db)
        except (OSError, ValueError):
            result["collection_audit"] = {
                "mode": "unavailable", "reason": "No readable actual-news database",
            }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
