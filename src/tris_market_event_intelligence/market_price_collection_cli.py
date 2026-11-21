"""Fetch observed historical adjusted daily closes for explicitly mapped instruments.

Optional research adapter for Yahoo Finance through yfinance. This is not a
point-in-time historical data archive or a guarantee of data availability.
No automatic instrument guesses or synthesized prices are permitted.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
from datetime import UTC, date, datetime
from math import isfinite
from pathlib import Path

from tris_market_event_intelligence.market_outcome_validation_cli import _mappings


def collect_prices(
    mappings_csv: Path, *, start: date, end: date, client: object,
) -> tuple[list[dict[str, str]], dict]:
    if start >= end:
        raise ValueError("Price collection interval must have start < end (exclusive)")
    mappings = _mappings(mappings_csv)
    if not mappings:
        raise ValueError("At least one verified target mapping is required")
    symbols = sorted({
        instrument for entry in mappings.values()
        for instrument in (entry["symbol"], entry["benchmark"])
    })
    observations: dict[tuple[str, str], dict[str, str]] = {}
    missing: list[str] = []
    for symbol in symbols:
        try:
            # auto_adjust=True means history Close has Yahoo's adjusted-close
            # basis; only interval=1d is admitted for this daily study.
            frame = client.Ticker(symbol).history(
                start=start.isoformat(), end=end.isoformat(),
                interval="1d", auto_adjust=True, actions=False,
                raise_errors=True,
            )
        except Exception as exc:  # noqa: BLE001 - third-party provider exception boundary
            # Do not substitute values from a different ticker or provider.
            missing.append(symbol)
            print(json.dumps({
                "symbol": symbol, "status": "SOURCE_ERROR",
                "errorType": type(exc).__name__,
            }), flush=True)
            continue
        if frame is None or frame.empty or "Close" not in frame:
            missing.append(symbol)
            continue
        for timestamp, close in frame["Close"].items():
            day = (timestamp.date() if isinstance(timestamp, datetime) else timestamp).isoformat()
            if not start.isoformat() <= day < end.isoformat():
                continue
            price = float(close)
            if not isfinite(price) or price <= 0:
                raise ValueError(f"Invalid adjusted price for {symbol} {day}")
            key = symbol, day
            if key in observations:
                raise ValueError("Duplicate price date returned by data source")
            observations[key] = {
                "symbol": symbol, "date": day,
                "adjusted_close": format(price, ".12g"),
                "source": "Yahoo Finance via yfinance; daily auto_adjust=True",
            }
    rows = [observations[key] for key in sorted(observations)]
    manifest = {
        "schemaVersion": "market_price_collection_v1",
        "provider": "Yahoo Finance via yfinance",
        "retrievedAt": datetime.now(UTC).isoformat(),
        "adjustment": "yfinance history(auto_adjust=True) Close",
        "periodStartInclusive": start.isoformat(),
        "periodEndExclusive": end.isoformat(),
        "requestedSymbols": symbols,
        "missingSymbols": missing,
        "priceRows": len(rows),
        "researchOnly": True,
        "pointInTimeArchive": False,
    }
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mappings", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.output.with_name(args.output.name + ".manifest.json")
    if any(path.exists() or path.is_symlink() for path in (args.output, manifest_path)):
        parser.error("Output or manifest exists: refusing to overwrite prior observations")
    try:
        yfinance = importlib.import_module("yfinance")
    except ImportError:
        parser.error("Optional dependency missing: python -m pip install yfinance")
    try:
        rows, manifest = collect_prices(
            args.mappings, start=args.start, end=args.end, client=yfinance,
        )
        if not rows:
            parser.error("No observations available; no file created")
        # Create-only artifacts; do not replace existing observations.
        with args.output.open("x", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(
                output, fieldnames=["symbol", "date", "adjusted_close", "source"],
            )
            writer.writeheader()
            writer.writerows(rows)
        with manifest_path.open("x", encoding="utf-8") as output:
            json.dump(manifest, output, indent=2, ensure_ascii=False)
            output.write("\n")
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"market-price-collection: {exc}\n")


if __name__ == "__main__":
    main()
