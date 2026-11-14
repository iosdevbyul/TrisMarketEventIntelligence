"""Offline source-pinned article-to-minute-price coverage audit, no trading."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from tris_market_event_intelligence.news_price_coverage import inspect_news_price_coverage


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare unapproved news issuer candidates to local minute source coverage"
    )
    parser.add_argument("--review-bundle", type=Path, required=True)
    parser.add_argument("--minute-manifest", type=Path, required=True)
    parser.add_argument("--benchmark", action="append", default=[],
                        help="Explicit exchange:symbol, e.g. XNAS:SPY")
    parser.add_argument("--as-of", required=True, help="Timezone-aware ISO 8601 cutoff")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        mappings = {}
        for item in args.benchmark:
            if item.count(":") != 1:
                raise ValueError("Expected --benchmark XNAS:SPY")
            market, symbol = item.split(":")
            if market in mappings:
                raise ValueError("Repeated benchmark market")
            mappings[market] = symbol
        result = inspect_news_price_coverage(
            args.review_bundle, args.minute_manifest,
            benchmark_symbols=mappings, as_of=datetime.fromisoformat(args.as_of),
        )
        if args.output is None:
            print(json.dumps(result, ensure_ascii=False))
        else:
            if args.output.is_symlink():
                raise ValueError("Output symlinks prohibited")
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            print(json.dumps({
                "status": result["status"],
                "report": str(args.output),
                "candidate_rows": result.get("candidates_inspected"),
                "return_labels_generated": 0,
            }))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.exit(2, f"news-price-coverage: {exc}\n")


if __name__ == "__main__":
    main()
