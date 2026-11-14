"""Opt-in operator CLI for append-only new-news economic event extraction."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from tris_market_event_intelligence.post_snapshot_materialization import (
    materialize_post_snapshot_events,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explicit append-only extraction for articles received after issuer capture",
    )
    parser.add_argument("--apply", action="store_true",
                        help="Required: authorize append-only event/checkpoint writes")
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--news-db", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    if not args.apply:
        parser.error("Explicit --apply required; default validation is read-only")
    try:
        report = materialize_post_snapshot_events(
            args.bundle.expanduser(), args.news_db.expanduser(),
            args.output_dir.expanduser(), batch_size=args.batch_size,
        )
    except ValueError as exc:
        parser.exit(2, f"post-snapshot-extraction: {str(exc)[:160]}\n")
    except (OSError, sqlite3.Error, TypeError, KeyError) as exc:
        parser.exit(2, f"post-snapshot-extraction: {type(exc).__name__}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
