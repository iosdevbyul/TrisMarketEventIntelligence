"""Opt-in portal HTML collection with robots enforcement and bounded rate."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from tris_market_event_intelligence.collector import NewsStore
from tris_market_event_intelligence.web_scraper import PORTALS, WebScrapingCollector


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape permitted Korean portal news HTML")
    parser.add_argument(
        "--db", type=Path,
        default=Path(os.getenv("MARKET_EVENT_DB", "data/news.sqlite3")),
    )
    parser.add_argument("--portals", default="naver,daum,nate")
    parser.add_argument("--max-articles", type=int, default=10)
    parser.add_argument("--delay", type=float, default=3.0)
    args = parser.parse_args()
    portals = tuple(x.strip() for x in args.portals.split(","))
    unknown = set(portals).difference(PORTALS)
    if unknown:
        parser.error(f"Unknown portals: {', '.join(sorted(unknown))}")
    collector = WebScrapingCollector(
        NewsStore(args.db), max_articles=args.max_articles,
        minimum_delay=args.delay,
    )
    result = collector.collect_once(portals)
    print(json.dumps({key: asdict(value) for key, value in result.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
