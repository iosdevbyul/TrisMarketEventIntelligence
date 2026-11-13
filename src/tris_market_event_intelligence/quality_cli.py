"""Generate local manual review sheets, or score a fully labelled one."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tris_market_event_intelligence.quality_audit import (
    create_review_sheet,
    score_review_sheet,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent local article quality audit")
    parser.add_argument(
        "--db", type=Path,
        default=Path(os.getenv("MARKET_EVENT_DB", "data/news.sqlite3")),
    )
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--export", type=Path, help="Write private CSV review sheet")
    actions.add_argument("--score", type=Path, help="Read labelled review sheet")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261113)
    args = parser.parse_args()
    if args.export:
        report = create_review_sheet(
            args.db, args.export, count=args.count, seed=args.seed,
        )
    else:
        report = score_review_sheet(args.db, args.score)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
