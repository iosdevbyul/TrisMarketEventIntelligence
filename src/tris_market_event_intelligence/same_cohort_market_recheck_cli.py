"""Re-evaluate real automatic news returns and compare only the SAME source impacts.

No model calls, quote downloads, database mutations, or invented market bars.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from tris_market_event_intelligence.automatic_gemma_market_outcomes_cli import (
    evaluate_automatic_outcomes,
)
from tris_market_event_intelligence.market_outcome_validation_cli import _mappings
from tris_market_event_intelligence.market_price_recovery_cli import _observations

MAX_REPORT_BYTES = 40 * 1024 * 1024
HORIZONS = (1, 5)


def _read_report(path: Path) -> dict:
    if (not path.is_file() or path.is_symlink()
            or path.stat().st_size > MAX_REPORT_BYTES):
        raise ValueError("Existing bounded immutable source report required")
    report = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(report, dict)
            or report.get("schemaVersion") != "automatic_gemma_daily_outcomes_v1"
            or report.get("sourceType") != "REAL_AUTOMATIC_SQLITE_GEMMA_LEDGER"
            or report.get("trainingEligible") is not False
            or report.get("prospectivePredictionAccuracy") is not None
            or report.get("researchOnly") is not True):
        raise ValueError("Expected retrospective real SQLite Gemma outcome report")
    return report


def _key(row: dict) -> tuple:
    article = row.get("articleId")
    index = row.get("impactIndex")
    digest = row.get("predictionSourceSha256")
    target = row.get("target")
    prediction = row.get("prediction")
    if (type(article) is not int or article < 1
            or type(index) is not int or index < 0
            or not isinstance(digest, str) or len(digest) != 64
            or not isinstance(target, str) or not target
            or prediction not in {"UP", "DOWN", "NEUTRAL", "UNCERTAIN"}
            or row.get("predictionSealedBeforeOutcome") is not False):
        raise ValueError("Invalid original source impact identity or unsealed state")
    return article, index, digest, target, prediction


def _by_identity(report: dict) -> dict[tuple, dict]:
    impacts = report.get("items")
    if (not isinstance(impacts, list)
            or len(impacts) != report.get("impactsInspected")):
        raise ValueError("Incomplete original impact report")
    indexed = {}
    for impact in impacts:
        if not isinstance(impact, dict):
            raise TypeError("Non-object impact row")
        key = _key(impact)
        if key in indexed:
            raise ValueError("Duplicate impact source identity")
        indexed[key] = impact
    return indexed


def _horizon(row: dict, n: int) -> dict | None:
    for item in row.get("horizons", []):
        if item.get("sessions") == n:
            return item
    return None


def compare_same_cohort(
    previous: dict, current: dict, mappings: Path, prices: Path,
) -> dict:
    before = _by_identity(previous)
    after = _by_identity(current)
    common = before.keys() & after.keys()
    mapped = _mappings(mappings)
    bars = _observations(prices)
    per_symbol = {
        symbol: {
            "firstObservedDay": rows[0]["date"],
            "lastObservedDay": rows[-1]["date"],
            "rows": len(rows),
        }
        for symbol, rows in sorted(bars.items()) if rows
    }
    transitions: Counter[str] = Counter()
    horizon_results = {}
    pending = []
    for key in sorted(common):
        old, new = before[key], after[key]
        transitions[f"{old['status']} -> {new['status']}"] += 1
        for horizon in HORIZONS:
            prior = _horizon(old, horizon)
            updated = _horizon(new, horizon)
            old_verdict = prior.get("verdict") if prior else "UNAVAILABLE"
            new_verdict = updated.get("verdict") if updated else "UNAVAILABLE"
            counter = horizon_results.setdefault(horizon, Counter())
            if old_verdict in {"CORRECT", "INCORRECT"}:
                counter["beforeScored"] += 1
            if new_verdict in {"CORRECT", "INCORRECT"}:
                counter["afterScored"] += 1
            if (old_verdict not in {"CORRECT", "INCORRECT"}
                    and new_verdict in {"CORRECT", "INCORRECT"}):
                counter["newlyScored"] += 1
            if (old_verdict in {"CORRECT", "INCORRECT"}
                    and new_verdict not in {"CORRECT", "INCORRECT"}):
                counter["lostScores"] += 1
        if (new.get("status") in {"NO_ENTRY_SESSION", "PARTIAL"}
                or any(h.get("dataStatus") == "INSUFFICIENT_FUTURE"
                       for h in new.get("horizons", []))):
            target = new["target"]
            market = new.get("mapping") or mapped.get(target)
            instrument = market.get("symbol") if market else None
            benchmark = market.get("benchmark") if market else None
            stock_range = per_symbol.get(instrument) if instrument else None
            benchmark_range = per_symbol.get(benchmark) if benchmark else None
            pending.append({
                "articleId": new["articleId"],
                "impactIndex": new["impactIndex"],
                "target": target, "prediction": new["prediction"],
                "publishedAt": new.get("publishedAt"),
                "status": new["status"], "entryDay": new.get("entryDay"),
                "instrument": instrument, "benchmark": benchmark,
                "lastInstrumentDay": (
                    stock_range["lastObservedDay"] if stock_range else None
                ),
                "lastBenchmarkDay": (
                    benchmark_range["lastObservedDay"] if benchmark_range else None
                ),
                "horizons": [
                    {
                        "sessions": h["sessions"],
                        "dataStatus": h["dataStatus"],
                        "exitDay": h.get("exitDay"),
                    }
                    for h in new.get("horizons", [])
                ],
                "reason": (
                    "NO_POST_PUBLICATION_CLOSE_IN_CURRENT_FILE"
                    if new["status"] == "NO_ENTRY_SESSION"
                    else "LATER_SESSION_OR_BENCHMARK_PRICE_NOT_YET_OBSERVED"
                ),
                "nextSourceAction": (
                    "Collect only real completed later sessions; do not "
                    "invent future closes or infer a model failure"
                ),
            })
    return {
        "schemaVersion": "same_cohort_market_maturity_v1",
        "previousImpactCount": len(before),
        "currentImpactCount": len(after),
        "commonIdenticalSourceImpacts": len(common),
        "newSourceImpacts": len(after.keys() - before.keys()),
        "removedOrChangedSourceImpacts": len(before.keys() - after.keys()),
        "cohortsIdentical": before.keys() == after.keys(),
        "statusTransitionsOnCommonCohort": dict(sorted(transitions.items())),
        "sameSourceHorizonScores": {
            f"{n}d": {
                "before": horizon_results[n]["beforeScored"],
                "after": horizon_results[n]["afterScored"],
                "newlyScored": horizon_results[n]["newlyScored"],
                "lostScores": horizon_results[n]["lostScores"],
            }
            for n in HORIZONS
        },
        "priceCoverageBySymbol": per_symbol,
        "notYetMatured": pending,
        "notYetMaturedCount": len(pending),
        "previousGlobalScoredDenominatorsNotComparableWithoutSameCohort": True,
        "prospectivePredictionAccuracy": None,
        "trainingEligible": False,
        "noSyntheticFuturePrices": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only same-cohort price recheck and maturity report",
    )
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--mappings", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--price-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--current-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if (args.output.resolve() == args.current_output.resolve()
                or any(path.exists() or path.is_symlink()
                       for path in (args.output, args.current_output))):
            raise ValueError("Create-only distinct output paths required")
        previous = _read_report(args.previous)
        current = evaluate_automatic_outcomes(
            args.db, args.mappings, args.prices, model=args.model,
            price_source="Existing source-provided adjusted daily price snapshot",
            price_manifest=args.price_manifest,
        )
        comparison = compare_same_cohort(
            previous, current, args.mappings, args.prices,
        )
        comparison.update({
            "previousReportSha256": hashlib.sha256(
                args.previous.read_bytes()
            ).hexdigest(),
            "currentPriceSha256": current["pricesSha256"],
            "currentMappingsSha256": current["mappingsSha256"],
            "currentReportPath": str(args.current_output),
            "productionDatabaseModified": False,
        })
        with args.current_output.open("x", encoding="utf-8") as handle:
            json.dump(current, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(comparison, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(json.dumps({
            "previous": comparison["previousImpactCount"],
            "current": comparison["currentImpactCount"],
            "sameSource": comparison["commonIdenticalSourceImpacts"],
            "newSource": comparison["newSourceImpacts"],
            "changedOrAbsent": comparison["removedOrChangedSourceImpacts"],
            "oneDay": comparison["sameSourceHorizonScores"]["1d"],
            "fiveDay": comparison["sameSourceHorizonScores"]["5d"],
            "notYetMatured": comparison["notYetMaturedCount"],
            "noNewPricesFetched": True,
            "output": str(args.output),
        }, ensure_ascii=False))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.exit(2, f"same-cohort-market-recheck: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
