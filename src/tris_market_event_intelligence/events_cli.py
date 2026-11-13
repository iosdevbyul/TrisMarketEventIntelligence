"""Batch event extraction CLI."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from tris_market_event_intelligence.event_store import EventProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract verified policy events from RSS")
    parser.add_argument("--db", type=Path, default=Path(
        os.getenv("MARKET_EVENT_DB", "data/news.sqlite3")
    ))
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()
    stats = EventProcessor(args.db).process_pending(args.batch_size)
    print(json.dumps(asdict(stats)))


if __name__ == "__main__":
    main()
