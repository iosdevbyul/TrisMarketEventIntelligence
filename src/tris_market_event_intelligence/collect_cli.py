"""News collector CLI, designed for one-shot or long-running launchd execution."""

import argparse
import os
from pathlib import Path

from tris_market_event_intelligence.collector import run_collector


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect official financial news RSS feeds")
    parser.add_argument("--once", action="store_true", help="Fetch each feed once and exit")
    parser.add_argument(
        "--db", type=Path,
        default=Path(os.getenv("MARKET_EVENT_DB", "data/news.sqlite3")),
    )
    parser.add_argument("--interval", type=int, default=300, help="Polling interval (seconds)")
    args = parser.parse_args()
    run_collector(args.db, args.interval, args.once)


if __name__ == "__main__":
    main()
