"""Recover previously discovered SEC filings, without inference or scheduling."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

from tris_market_event_intelligence.source_article_content import collect_full_text


def coverage(path: Path) -> list[dict]:
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("PRAGMA query_only=ON")
        return [
            {"source": source, "status": status, "errorCode": error, "count": count}
            for source, status, error, count in db.execute("""
                SELECT a.source,COALESCE(f.status,'NOT_CAPTURED'),f.error_code,COUNT(*)
                FROM articles a LEFT JOIN article_full_text f ON a.id=f.article_id
                WHERE a.source GLOB 'sec_8k_[0-9]*'
                GROUP BY a.source,COALESCE(f.status,'NOT_CAPTURED'),f.error_code
                ORDER BY a.source,COALESCE(f.status,'NOT_CAPTURED'),f.error_code
            """)
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12, choices=range(1, 13))
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.db.is_file() or args.db.is_symlink():
        parser.error("Existing non-symlink DB required")
    result = None
    if args.run:
        if "@" not in os.environ.get("MARKET_EVENT_SEC_USER_AGENT", ""):
            parser.error("Set MARKET_EVENT_SEC_USER_AGENT to your actual name and contact email")
        result = collect_full_text(args.db, limit=args.limit, sec_only=True)
    print(json.dumps({
        "capture": result, "coverage": coverage(args.db),
        "modelCalls": 0, "marketReturnsAnalyzed": False,
        "historicalPublicTimesIndependentlyVerified": False,
    }, indent=2))


if __name__ == "__main__":
    main()
