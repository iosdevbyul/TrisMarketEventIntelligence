"""Explicit offline historical NEWS + price + macro research packet builder."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tris_market_event_intelligence.historical_shock_research import (
    build_historical_shock_research,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build non-causal historical TOP shock and control study"
    )
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--benchmark-symbol", required=True)
    parser.add_argument("--stock-csv", type=Path, required=True)
    parser.add_argument("--benchmark-csv", type=Path, required=True)
    parser.add_argument("--news-cohort", type=Path, required=True)
    parser.add_argument("--macro-csv", type=Path, required=True)
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--controls", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preview", action="store_true",
                        help="Audit source inputs and counts without saving a packet")
    args = parser.parse_args()
    try:
        if not args.preview and args.output is None:
            raise ValueError("Require --output for private research artifact")
        if args.output is not None and (
            args.output.is_symlink() or args.output.exists()
            or not args.output.parent.is_dir()
            or args.output.parent.is_symlink()
        ):
            raise ValueError("Only create a new output under an existing directory")
        report = build_historical_shock_research(
            symbol=args.symbol, benchmark_symbol=args.benchmark_symbol,
            stock_path=args.stock_csv, benchmark_path=args.benchmark_csv,
            news_cohort_path=args.news_cohort, macro_path=args.macro_csv,
            top=args.top, controls=args.controls,
        )
        if args.output is not None and not args.preview:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(args.output, flags, 0o600)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    json.dump(report, stream, ensure_ascii=False, indent=2)
                    stream.write("\n")
            except BaseException:
                args.output.unlink(missing_ok=True)
                raise
        print(json.dumps({
            "status": report["status"],
            "symbol": report["symbol"],
            "topCount": len(report["topEvents"]),
            "controlCount": len(report["controlEvents"]),
            "newsObservedDatesUTC": report["newsObservedDatesUTC"],
            "macroSourceAuthenticityVerified": False,
            "priceSourceAuthenticityVerified": False,
            "trainingDataReady": False,
            "llmCalls": 0,
            "outputCreated": bool(args.output is not None and not args.preview),
        }, ensure_ascii=False))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"historical-shock-research: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
