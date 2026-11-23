"""Source-backed directly named equities: real adjusted-price coverage bridge.

The two exact names here originate from the user's real automatic Gemma
outcome report. A documented listing does NOT authenticate news timing,
causation, price provenance, or a pre-outcome forecast.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path

from tris_market_event_intelligence.automatic_gemma_market_outcomes_cli import (
    evaluate_automatic_outcomes,
)
from tris_market_event_intelligence.market_outcome_validation_cli import (
    _csv,
    _mappings,
    _timestamp,
)
from tris_market_event_intelligence.market_price_collection_cli import collect_prices
from tris_market_event_intelligence.market_price_recovery_cli import (
    FIELDS,
    _observations,
    recover_prices,
)
from tris_market_event_intelligence.market_proxy_cohort_cli import CSV_FIELDS

# Exact observed issuer names and original article IDs. Do not map generic
# sectors, an exchange venue, a policy institution, or a private corporation.
# URLs establish listed-security identity only; NO historical first-public proof.
EQUITY_CATALOG = {
    ("Nvidia", 20975): {
        "market": "XNAS", "symbol": "NVDA", "benchmark": "SPY",
        "evidence_url": "https://investor.nvidia.com/investor-resources/faqs/default.aspx",
    },
    ("Capita", 20989): {
        "market": "XLON", "symbol": "CPI.L", "benchmark": "ISF.L",
        "evidence_url": "https://www.londonstockexchange.com/en/quote/equity/CPI/capita-plc",
    },
}
MAX_BYTES = 30 * 1024 * 1024


def sha(path: Path) -> str:
    if (not path.is_file() or path.is_symlink()
            or path.stat().st_size > MAX_BYTES):
        raise ValueError("Original source must be an existing bounded regular file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_direct_equities(
    outcomes: Path, previous_mappings: Path, previous_prices: Path,
    *, verified_at: datetime,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict]:
    if verified_at.utcoffset() is None:
        raise ValueError("Listing verification time must be timezone-aware")
    original_sha = sha(outcomes)
    previous_map_sha = sha(previous_mappings)
    previous_price_sha = sha(previous_prices)
    result = json.loads(outcomes.read_text(encoding="utf-8"))
    if (not isinstance(result, dict)
            or result.get("schemaVersion") != "automatic_gemma_daily_outcomes_v1"
            or result.get("sourceType") != "REAL_AUTOMATIC_SQLITE_GEMMA_LEDGER"
            or result.get("researchOnly") is not True
            or result.get("trainingEligible") is not False
            or result.get("prospectivePredictionAccuracy") is not None):
        raise ValueError("Only real retrospective automatic ledger outcomes supported")
    impacts = result.get("items")
    if (not isinstance(impacts, list)
            or len(impacts) != result.get("impactsInspected")):
        raise ValueError("Incomplete source impact results")
    previous_rows = _csv(previous_mappings, set(CSV_FIELDS[:5]))
    _mappings(previous_mappings)
    if len(previous_rows) > 2000:
        raise ValueError("Excessive original mappings")
    original_symbols = _observations(previous_prices)
    preserved = [
        {field: row.get(field, "") or "" for field in CSV_FIELDS}
        for row in previous_rows
    ]
    existing_targets = {row["target"] for row in preserved}
    found: dict[str, dict] = {}
    excluded = []
    for item in impacts:
        name = item.get("target")
        article = item.get("articleId")
        if not isinstance(name, str) or type(article) is not int:
            raise ValueError("Invalid original impact identity")
        rule = EQUITY_CATALOG.get((name, article))
        if rule is None or name in existing_targets:
            continue
        reason = None
        if item.get("status") != "UNMAPPED":
            reason = "NOT_CURRENTLY_UNMAPPED"
        elif item.get("sourceAuditStatus") != "STORED_ORIGINAL_SHA_MATCHED":
            reason = "SOURCE_ORIGINAL_NOT_ADMITTED"
        elif item.get("prediction") not in ("UP", "DOWN"):
            reason = "NOT_A_DIRECTIONAL_FORECAST"
        elif item.get("predictionSealedBeforeOutcome") is not False:
            reason = "UNEXPECTED_PROSPECTIVE_CLAIM"
        elif not isinstance(item.get("publishedAt"), str):
            reason = "NO_SOURCE_PUBLICATION_CLOCK"
        else:
            _timestamp(item["publishedAt"])
        if reason:
            excluded.append({"target": name, "articleId": article, "reason": reason})
            continue
        if name in found:
            raise ValueError("Repeated direct issuer candidate in stored source")
        found[name] = {
            "articleId": article, "direction": item["prediction"],
            "publishedAt": item["publishedAt"], **rule,
        }

    if not found:
        raise ValueError("No source-admitted unscored named equity directions")
    for name, item in found.items():
        preserved.append({
            "target": name, "market": item["market"], "symbol": item["symbol"],
            "benchmark": item["benchmark"], "verified_at": verified_at.isoformat(),
            "mapping_kind": "DIRECT_EQUITY", "proxy_limitation": "",
            "evidence_url": item["evidence_url"],
        })
    preserved.sort(key=lambda row: row["target"])
    pending = [row for row in preserved if row["target"] in found]
    symbols = sorted({
        row["symbol"] for row in pending
    } | {row["benchmark"] for row in pending})
    # A fresh historical quote batch includes benchmark *even if* it was
    # present in an older snapshot, so same-era price windows can be observed.
    return preserved, pending, {
        "schemaVersion": "listed_equity_real_price_expansion_v1",
        "outcomesSha256": original_sha,
        "previousMappingsSha256": previous_map_sha,
        "previousPricesSha256": previous_price_sha,
        "sourceImpactCount": len(impacts),
        "newDirectEquities": len(found),
        "candidates": found,
        "excluded": excluded,
        "quoteSymbols": symbols,
        "existingPriceCounts": {
            symbol: len(original_symbols.get(symbol, [])) for symbol in symbols
        },
        "newRealPricesFetched": False,
        "newDirectionScoresObserved": None,
        "mappingVerifiedAt": verified_at.isoformat(),
        "listingEvidenceDoesNotProveEventConnection": True,
        "forecastSealedBeforeOutcome": False,
        "trainingEligible": False,
        "prospectivePredictionAccuracy": None,
        "notes": (
            "Explicit exact issuer+article direction candidate, no sector or "
            "macro ETF guessing. New listing review is retrospective. Daily "
            "return begins at first session strictly after publication day, "
            "NOT at first market reaction. Future 1/5-session bars cannot be "
            "scored before actually observed."
        ),
    }


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(
    outcomes: Path, db: Path, previous_mappings: Path, previous_prices: Path,
    output_dir: Path, *, model: str, fetch: bool = False,
    start: date | None = None, end: date | None = None,
) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("Output directory already exists: never overwrite research")
    if not output_dir.parent.is_dir() or output_dir.parent.is_symlink():
        raise ValueError("Parent output folder must already exist")
    now = datetime.now(UTC)
    combined, quotes_map, plan = prepare_direct_equities(
        outcomes, previous_mappings, previous_prices, verified_at=now,
    )
    if fetch and (start is None or end is None or start >= end
                  or end > now.date()):
        raise ValueError("Explicit historical period must exclude the current unfinished UTC date")
    # Assemble in an isolated same-filesystem folder; final name appears only
    # after all validations and real price observations finish successfully.
    with tempfile.TemporaryDirectory(
        prefix=".listed-price-build-", dir=output_dir.parent,
    ) as temp:
        tmp = Path(temp)
        mapping_path = tmp / "reviewed-mappings.csv"
        subset_path = tmp / "new-listed-quote-targets.csv"
        write_csv(mapping_path, CSV_FIELDS, combined)
        write_csv(subset_path, CSV_FIELDS, quotes_map)
        if fetch:
            yfinance = importlib.import_module("yfinance")
            prices, collection = collect_prices(
                subset_path, start=start, end=end, client=yfinance,
            )
            if not prices:
                raise ValueError("Provider returned no actual adjusted daily prices")
            fetched = tmp / "source-adjusted-daily.csv"
            write_csv(fetched, FIELDS, prices)
            collection["outputSha256"] = sha(fetched)
            (tmp / "source-adjusted-daily.manifest.json").write_text(
                json.dumps(collection, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            # Whole-instrument priority: newly fetched complete instrument
            # series first; old instruments only when wholly absent from fresh
            # source. Never splice conflicting points by date.
            combined_prices, recovery = recover_prices(
                fetched, previous_prices, mapping_path,
            )
            if recovery["missingSymbols"]:
                raise ValueError(
                    f"No actual prices for required symbols: {recovery['missingSymbols']}"
                )
            expanded = tmp / "expanded-adjusted-daily.csv"
            write_csv(expanded, FIELDS, combined_prices)
            recovery["outputSha256"] = sha(expanded)
            with (tmp / "expanded-adjusted-daily.csv.manifest.json").open(
                "x", encoding="utf-8"
            ) as f:
                json.dump(recovery, f, ensure_ascii=False, indent=2)
                f.write("\n")
            evaluated = evaluate_automatic_outcomes(
                db, mapping_path, expanded, model=model,
                price_source="Yahoo Finance adjusted daily + preserved whole-symbol history",
                price_manifest=tmp / "expanded-adjusted-daily.csv.manifest.json",
            )
            with (tmp / "automatic-observed-outcomes.json").open(
                "x", encoding="utf-8"
            ) as f:
                json.dump(evaluated, f, ensure_ascii=False, indent=2)
                f.write("\n")
            old = json.loads(outcomes.read_text(encoding="utf-8"))
            plan["newRealPricesFetched"] = True
            plan["quoteRows"] = len(prices)
            plan["providerMissingSymbols"] = collection["missingSymbols"]
            plan["scoredDirectionBefore"] = old["metrics"]["1d"]["denominator"]
            plan["scoredDirectionAfter"] = evaluated["metrics"]["1d"]["denominator"]
            plan["newDirectionScoresObserved"] = (
                plan["scoredDirectionAfter"] - plan["scoredDirectionBefore"]
            )
            plan["newDailyObservedImpacts"] = (
                evaluated["statusCounts"].get("COMPLETED", 0) -
                old["statusCounts"].get("COMPLETED", 0)
            )
            plan["evaluatedImpactCount"] = evaluated["impactsInspected"]
            plan["historicalPriceCoverageIncomplete"] = True
        with (tmp / "coverage-plan.json").open("x", encoding="utf-8") as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)
            f.write("\n")
        # tempfile is removed in its context. Rename before leaving the
        # context so its now-absent source causes no data corruption.
        tmp.rename(output_dir)
        # rename may make context-manager cleanup on old path a no-op.
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Real named equities → historical price coverage")
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--mappings", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--fetch-prices", action="store_true")
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    args = parser.parse_args()
    try:
        result = run(
            args.outcomes, args.db, args.mappings, args.prices,
            args.output_dir, model=args.model,
            fetch=args.fetch_prices, start=args.start, end=args.end,
        )
        print(json.dumps({
            "status": "COMPLETED", "newDirectEquities": result["newDirectEquities"],
            "quoteSymbols": result["quoteSymbols"],
            "newRealPricesFetched": result["newRealPricesFetched"],
            "newDirectionScoresObserved": result["newDirectionScoresObserved"],
            "outputDir": str(args.output_dir),
        }, ensure_ascii=False))
    except (ValueError, TypeError, KeyError, OSError, ImportError) as exc:
        parser.exit(2, f"listed-equity-market-prices: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
