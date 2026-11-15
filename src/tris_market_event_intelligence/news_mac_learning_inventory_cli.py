"""Summarize local private NEWS research prerequisites without raw source text."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tris_market_event_intelligence.news_mac_learning_inventory import (
    inspect_mac_news_learning,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only Mac NEWS learning inventory (summary-only stdout)"
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()
    try:
        result = inspect_mac_news_learning(data_dir=args.data_dir)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.exit(2, f"news-mac-learning-inventory: {type(exc).__name__}: {exc}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
