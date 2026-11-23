"""Recover completely missing instruments from a preserved older real-price export.

Never mix two provider snapshots *within* an instrument. A missing symbol in
the primary file may use all observations from a preserved fallback snapshot.
Do not fabricate rows, merge conflicting records, or claim full time coverage.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import UTC, date, datetime
from math import isfinite
from pathlib import Path

from tris_market_event_intelligence.market_outcome_validation_cli import _csv, _mappings

FIELDS = ["symbol", "date", "adjusted_close", "source"]


def _observations(path: Path) -> dict[str, list[dict[str, str]]]:
    values: dict[str, list[dict[str, str]]] = {}
    seen: set[tuple[str, str]] = set()
    for row in _csv(path, set(FIELDS)):
        symbol, day, origin = (
            row["symbol"].strip(), row["date"].strip(), row["source"].strip()
        )
        if not symbol or not origin:
            raise ValueError("Each price observation requires symbol and source")
        date.fromisoformat(day)
        price = float(row["adjusted_close"])
        if not isfinite(price) or price <= 0:
            raise ValueError("Price must be positive and finite")
        key = (symbol, day)
        if key in seen:
            raise ValueError(f"Duplicate price observation: {symbol} {day}")
        seen.add(key)
        values.setdefault(symbol, []).append({
            "symbol": symbol, "date": day,
            "adjusted_close": row["adjusted_close"].strip(),
            "source": origin,
        })
    for group in values.values():
        group.sort(key=lambda row: row["date"])
    return values


def recover_prices(
    primary: Path, fallback: Path, mappings: Path,
) -> tuple[list[dict[str, str]], dict]:
    """Include fallback only for instruments wholly absent from primary."""
    if primary.resolve() == fallback.resolve():
        raise ValueError("Primary and fallback cannot be the same file")
    mapped = _mappings(mappings)
    required = sorted({
        instrument for row in mapped.values()
        for instrument in (row["symbol"], row["benchmark"])
    })
    latest = _observations(primary)
    earlier = _observations(fallback)
    unexpected = (set(latest) | set(earlier)) - set(required)
    if unexpected:
        raise ValueError(f"Unexpected symbols in source observations: {sorted(unexpected)}")
    recovered = []
    missing = []
    chosen: dict[str, list[dict[str, str]]] = {}
    origins: dict[str, str] = {}
    for symbol in required:
        if symbol in latest:
            chosen[symbol] = latest[symbol]
            origins[symbol] = "primary"
        elif symbol in earlier:
            chosen[symbol] = earlier[symbol]
            origins[symbol] = "fallback"
            recovered.append(symbol)
        else:
            missing.append(symbol)
    rows = sorted(
        (row for series in chosen.values() for row in series),
        key=lambda row: (row["symbol"], row["date"]),
    )
    manifest = {
        "schemaVersion": "market_price_recovery_v1",
        "createdAt": datetime.now(UTC).isoformat(),
        "primarySha256": hashlib.sha256(primary.read_bytes()).hexdigest(),
        "fallbackSha256": hashlib.sha256(fallback.read_bytes()).hexdigest(),
        "mappingsSha256": hashlib.sha256(mappings.read_bytes()).hexdigest(),
        "policy": "One intact source file per instrument; fallback only for fully absent symbols",
        "requestedSymbols": required,
        "symbolOrigin": origins,
        "recoveredSymbols": recovered,
        "missingSymbols": missing,
        "symbolRowCounts": dict(sorted(Counter(
            row["symbol"] for row in rows
        ).items())),
        "priceRows": len(rows),
        "completeInstrumentCoverage": not missing,
        "completeDateCoverageGuaranteed": False,
        "researchOnly": True,
        "pointInTimeArchive": False,
    }
    return rows, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--mappings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.output.with_name(args.output.name + ".manifest.json")
    if any(path.exists() or path.is_symlink() for path in (args.output, manifest_path)):
        parser.error("Output already exists: refusing to overwrite source evidence")
    try:
        rows, manifest = recover_prices(args.primary, args.fallback, args.mappings)
        if manifest["missingSymbols"]:
            raise ValueError(f"Missing required symbols: {manifest['missingSymbols']}")
        with args.output.open("x", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        manifest["outputSha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
        with manifest_path.open("x", encoding="utf-8") as output:
            json.dump(manifest, output, ensure_ascii=False, indent=2)
            output.write("\n")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"market-price-recovery: {exc}\n")


if __name__ == "__main__":
    main()
