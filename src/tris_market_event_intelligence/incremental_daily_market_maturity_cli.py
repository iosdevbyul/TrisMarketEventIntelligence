"""Incremental observed adjusted-daily price extension and SAME-source Gemma rescoring.

Only an explicit user invocation fetches new bars. Prior observations
and reports remain immutable. Provider restatements of overlapping
adjusted closes fail closed: no cross-snapshot value splicing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import tempfile
from datetime import UTC, date, datetime
from math import isclose
from pathlib import Path

from tris_market_event_intelligence.automatic_gemma_market_outcomes_cli import (
    evaluate_automatic_outcomes,
)
from tris_market_event_intelligence.market_outcome_validation_cli import _mappings
from tris_market_event_intelligence.market_price_collection_cli import collect_prices
from tris_market_event_intelligence.market_price_recovery_cli import FIELDS, _observations
from tris_market_event_intelligence.same_cohort_market_recheck_cli import (
    _read_report,
    compare_same_cohort,
)

MAX_SOURCE_BYTES = 30 * 1024 * 1024


def _sha(path: Path) -> str:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError("Expected a bounded regular input file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extend_observed_daily(
    previous_prices: Path, incremental: list[dict[str, str]],
    required_symbols: set[str],
) -> tuple[list[dict[str, str]], dict]:
    """Require one overlapping real close per symbol and no old-bar changes."""
    old = _observations(previous_prices)
    if not required_symbols or not required_symbols <= old.keys():
        raise ValueError("Every requested instrument needs an existing observed anchor")
    supplied: dict[str, list[dict[str, str]]] = {}
    for item in incremental:
        symbol = item["symbol"]
        if symbol not in required_symbols:
            raise ValueError(f"Unrequested instrument in provider response: {symbol}")
        supplied.setdefault(symbol, []).append(item)
    added: dict[str, int] = {}
    merged = {symbol: list(rows) for symbol, rows in old.items()}
    conflicts = []
    for symbol in sorted(required_symbols):
        original = old[symbol]
        anchor = original[-1]
        if symbol not in supplied:
            raise ValueError(f"No fresh provider series for requested symbol {symbol}")
        observations = sorted(supplied[symbol], key=lambda item: item["date"])
        seen = set()
        overlap = False
        additions = []
        for row in observations:
            day = date.fromisoformat(row["date"])
            if day.isoformat() in seen:
                raise ValueError(f"Provider duplicated {symbol} {day}")
            seen.add(day.isoformat())
            price = float(row["adjusted_close"])
            if not 0 < price < float("inf") or not row["source"].strip():
                raise ValueError("Invalid observed price/source")
            if day.isoformat() <= anchor["date"]:
                existing = next(
                    (x for x in original if x["date"] == day.isoformat()), None
                )
                if existing is None:
                    raise ValueError("Provider supplied unanchored old date")
                if not isclose(
                    price, float(existing["adjusted_close"]),
                    rel_tol=0.000001, abs_tol=0.000001,
                ):
                    conflicts.append({"symbol": symbol, "date": day.isoformat()})
                if day.isoformat() == anchor["date"]:
                    overlap = True
            else:
                additions.append(row)
        if not overlap:
            raise ValueError(f"Missing exact most-recent date anchor for {symbol}")
        added[symbol] = len(additions)
        merged[symbol].extend(additions)
    if conflicts:
        raise ValueError(
            f"Adjusted closes revised across provider snapshots; full-source "
            f"refresh/revalidation required, no stitching: {conflicts}"
        )
    rows = sorted(
        (row for group in merged.values() for row in group),
        key=lambda item: (item["symbol"], item["date"]),
    )
    return rows, {
        "schemaVersion": "incremental_daily_price_extension_v1",
        "sourcePricesSha256": _sha(previous_prices),
        "requiredSymbols": sorted(required_symbols),
        "overlappingAnchorValidated": True,
        "overlapPriceTolerance": {"relative": 0.000001, "absolute": 0.000001},
        "newRowsBySymbol": added,
        "newPriceRows": sum(added.values()),
        "existingPriceRows": sum(len(group) for group in old.values()),
        "resultPriceRows": len(rows),
        "adjustmentRestatementBlocked": True,
        "freshProviderObservationsNotIndependentlyAuthenticated": True,
        "pointInTimeArchive": False,
        "trainingEligible": False,
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def refresh_and_recheck(
    previous_report: Path, news_db: Path, mappings: Path,
    previous_prices: Path, quote_targets: Path, output_dir: Path,
    *, model: str, end: date, client: object,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("Refusing to overwrite prior market observations")
    if not output_dir.parent.is_dir():
        raise ValueError("Output parent must exist")
    if end > datetime.now(UTC).date():
        raise ValueError("End date is exclusive: cannot include an unfinished UTC day")
    old_report = _read_report(previous_report)
    old_prices = _observations(previous_prices)
    quotes = _mappings(quote_targets)
    full = _mappings(mappings)
    for target, item in quotes.items():
        existing = full.get(target)
        if existing is None or any(
            item[field] != existing[field] for field in ("symbol", "benchmark", "market")
        ):
            raise ValueError(f"Quote target not in approved source mapping: {target}")
    symbols = {
        ticker for item in quotes.values()
        for ticker in (item["symbol"], item["benchmark"])
    }
    if not symbols or not symbols <= old_prices.keys():
        raise ValueError("Missing original price anchor for approved quote symbols")
    # One request for all quote-target symbols, anchored at oldest last
    # observed trading date. Never collect the prior 2026 Jan-Nov history.
    start = min(date.fromisoformat(old_prices[symbol][-1]["date"]) for symbol in symbols)
    if start >= end:
        raise ValueError("No completed calendar dates after the latest price anchor")
    new_rows, provider_manifest = collect_prices(
        quote_targets, start=start, end=end, client=client,
    )
    if provider_manifest["missingSymbols"]:
        raise ValueError(f"Provider missing requested instruments: {provider_manifest['missingSymbols']}")
    rows, extension = extend_observed_daily(previous_prices, new_rows, symbols)
    if extension["newPriceRows"] == 0:
        raise ValueError("No genuinely new observed daily closes to evaluate")
    # All snapshots and results are staged and published together.
    with tempfile.TemporaryDirectory(
        prefix=".incremental-market-", dir=output_dir.parent,
    ) as temp:
        staged = Path(temp)
        expanded = staged / "expanded-adjusted-daily.csv"
        _write_csv(expanded, rows)
        manifest = {
            "schemaVersion": "incremental_price_maturity_manifest_v1",
            "createdAt": datetime.now(UTC).isoformat(),
            "primarySha256": extension["sourcePricesSha256"],
            "mappingsSha256": _sha(mappings),
            "outputSha256": _sha(expanded),
            "provider": provider_manifest,
            "extension": extension,
            "periodEndExclusive": end.isoformat(),
            "completeDateCoverageGuaranteed": False,
            "pointInTimeArchive": False,
            "researchOnly": True,
        }
        (staged / "expanded-adjusted-daily.csv.manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        current = evaluate_automatic_outcomes(
            news_db, mappings, expanded, model=model,
            price_source="Yahoo adjusted historical daily with validated anchors",
        )
        comparison = compare_same_cohort(old_report, current, mappings, expanded)
        result = {
            **extension,
            "previousImpactCount": comparison["previousImpactCount"],
            "currentImpactCount": comparison["currentImpactCount"],
            "commonIdenticalSourceImpacts": comparison["commonIdenticalSourceImpacts"],
            "newSourceImpacts": comparison["newSourceImpacts"],
            "removedOrChangedSourceImpacts": comparison["removedOrChangedSourceImpacts"],
            "sameSourceHorizonScores": comparison["sameSourceHorizonScores"],
            "notYetMatured": comparison["notYetMatured"],
            "previousOutcomesSha256": _sha(previous_report),
            "newMarketOutcomeReportSha256": None,
            "prospectivePredictionAccuracy": None,
            "historicalUnsealedOnly": True,
            "productionDatabaseModified": False,
        }
        out = staged / "automatic-observed-outcomes.json"
        out.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result["newMarketOutcomeReportSha256"] = _sha(out)
        (staged / "same-cohort-coverage.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staged.rename(output_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--mappings", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--quote-targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    try:
        client = importlib.import_module("yfinance")
        result = refresh_and_recheck(
            args.previous, args.db, args.mappings, args.prices,
            args.quote_targets, args.output_dir, model=args.model, end=args.end,
            client=client,
        )
        print(json.dumps({
            "status": "COMPLETED",
            "newPriceRows": result["newPriceRows"],
            "newRowsBySymbol": result["newRowsBySymbol"],
            "commonIdenticalSourceImpacts": result["commonIdenticalSourceImpacts"],
            "newSourceImpacts": result["newSourceImpacts"],
            "oneDay": result["sameSourceHorizonScores"]["1d"],
            "fiveDay": result["sameSourceHorizonScores"]["5d"],
            "notYetMatured": len(result["notYetMatured"]),
            "prospectivePredictionAccuracy": None,
            "outputDir": str(args.output_dir),
        }, ensure_ascii=False))
    except (ValueError, TypeError, KeyError, ImportError, OSError) as exc:
        parser.exit(2, f"incremental-market-maturity: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
