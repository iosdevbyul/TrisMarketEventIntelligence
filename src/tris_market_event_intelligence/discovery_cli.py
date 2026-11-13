"""One-shot multi-source discovery and story indexing.

RSS collection is optional here so existing RSS CLI behaviour remains unchanged.
"""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from tris_market_event_intelligence.collector import NewsStore, OfficialNewsCollector
from tris_market_event_intelligence.discovery import SearchDiscovery
from tris_market_event_intelligence.source_catalog import load_feeds
from tris_market_event_intelligence.story_index import StoryIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover and group market-news leads")
    parser.add_argument(
        "--db", type=Path, default=Path(os.getenv("MARKET_EVENT_DB", "data/news.sqlite3"))
    )
    parser.add_argument("--include-rss", action="store_true")
    parser.add_argument("--feeds-file", type=Path)
    parser.add_argument("--skip-naver", action="store_true")
    parser.add_argument("--skip-gdelt", action="store_true")
    parser.add_argument("--index-only", action="store_true")
    args = parser.parse_args()
    if not args.index_only and args.skip_naver and args.skip_gdelt and not args.include_rss:
        parser.error("At least one source must be enabled")
    store = NewsStore(args.db)
    result: dict[str, object] = {}
    if not args.index_only:
        if args.include_rss:
            result["rss"] = OfficialNewsCollector(
                store, feeds=load_feeds(args.feeds_file)
            ).collect_once()
        discovered = SearchDiscovery(
            store,
            naver_client_id=os.getenv("NAVER_CLIENT_ID"),
            naver_client_secret=os.getenv("NAVER_CLIENT_SECRET"),
        ).collect_once(naver=not args.skip_naver, gdelt=not args.skip_gdelt)
        result["discovery"] = {k: asdict(v) for k, v in discovered.items()}
    index = StoryIndex(args.db)
    total = 0
    groups = 0
    matches = 0
    while True:
        progress = index.index_pending()
        total += progress.indexed
        groups += progress.new_groups
        matches += progress.url_matches + progress.title_matches
        if progress.indexed < 1000:
            break
    result["story_index"] = {
        "indexed": total, "new_groups": groups, "matched": matches,
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
