"""Inspect explicit human ticker review decisions; no DB or service changes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tris_market_event_intelligence.ticker_human_review import review_ticker_candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit pinned, reviewer-supplied event→ticker decisions offline",
    )
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = review_ticker_candidates(
            args.proposals.expanduser(), args.decisions.expanduser(),
        )
    except (OSError, ValueError, TypeError, KeyError, UnicodeError) as exc:
        parser.exit(2, f"ticker-human-review: {type(exc).__name__}\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
