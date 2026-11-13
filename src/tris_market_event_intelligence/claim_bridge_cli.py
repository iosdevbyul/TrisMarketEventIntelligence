"""Opt-in one-shot claim-to-event materialization, without LLM calls."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tris_market_event_intelligence.claim_event_bridge import ClaimEventBridge


def main() -> None:
    parser = argparse.ArgumentParser(description="Bridge collected claims into impact inputs")
    base = Path.home() / "Library" / "Application Support" / "TrisMarketEventIntelligence"
    parser.add_argument("--db", type=Path, default=base / "news.sqlite3")
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    try:
        result = ClaimEventBridge(args.db.expanduser()).process_pending(args.batch_size)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(asdict(result)))


if __name__ == "__main__":
    main()
