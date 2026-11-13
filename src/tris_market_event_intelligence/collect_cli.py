"""CLI for local, configurable financial news RSS collection."""

import argparse
import os
from pathlib import Path

from tris_market_event_intelligence.collector import run_collector
from tris_market_event_intelligence.source_catalog import load_feeds


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect financial news from RSS feeds")
    parser.add_argument("--once", action="store_true", help="Fetch selected feeds once")
    parser.add_argument("--list-sources", action="store_true", help="Show configured feed sources")
    parser.add_argument("--feeds-file", type=Path, help="Optional local JSON with additional RSS feeds")
    parser.add_argument("--only", help="Comma-separated IDs of feeds to collect")
    parser.add_argument(
        "--db", type=Path,
        default=Path(os.getenv("MARKET_EVENT_DB", "data/news.sqlite3")),
    )
    parser.add_argument("--interval", type=int, default=300, help="Polling interval (seconds)")
    args = parser.parse_args()
    ids = {part.strip() for part in args.only.split(",")} if args.only else None
    try:
        feeds = load_feeds(args.feeds_file, only=ids)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.list_sources:
        for source, url in feeds.items():
            print(f"{source}\t{url}")
        return
    run_collector(args.db, args.interval, args.once, feeds=feeds)


if __name__ == "__main__":
    main()
