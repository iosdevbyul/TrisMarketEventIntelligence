"""One-shot new-event forecast capture with actual later market observations.

This command may write NEW automatic Gemma analyses and opt-in forecast
precommitments to the given news DB. It never retroactively labels a
historical analysis as a live forecast. Existing adjusted-close prices are
read-only, and NO price provider is contacted during the forecast stage.

Every scored row is a *locally timestamped candidate*, not certified
prospective accuracy. True first-public source availability requires
independent authentication beyond a collector's publication metadata.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.automatic_news_pipeline import process
from tris_market_event_intelligence.committed_forecast_market_outcomes_cli import (
    evaluate as evaluate_commitments,
)


def cycle(
    db: Path, mappings: Path, prices: Path, output: Path, *,
    model: str = "gemma4:e2b", limit: int = 10, fresh_minutes: int = 120,
    collect: bool = True, price_manifest: Path | None = None,
    event_aware: bool = False,
) -> dict:
    if output.exists() or output.is_symlink():
        raise ValueError("Immutable result path must not exist")
    if not output.parent.is_dir():
        raise ValueError("Result output parent must exist")
    if not prices.is_file() or prices.is_symlink():
        raise ValueError("Existing real observed price CSV is required")
    # Price evaluation happens only after the model response and its
    # commitments are atomically persisted; there is no price access
    # inside model capture or model input construction.
    captured = process(
        db, model=model, limit=limit, collect=collect, verification_runs=1,
        seal_forecasts=True, mappings_path=mappings,
        fresh_only_minutes=fresh_minutes,
        **({"event_aware": True} if event_aware else {}),
    )
    if captured["status"] == "ALREADY_RUNNING":
        return {
            "status": "ALREADY_RUNNING", "modelInvoked": False,
            "priceEvaluationPerformed": False, "trainingEligible": False,
        }
    price_result = evaluate_commitments(
        db, prices, price_manifest,
    )
    summary = {
        "schemaVersion": "forward_news_market_cycle_v1",
        "runAt": datetime.now(UTC).isoformat(),
        "status": captured["status"],
        "newsCollectionStatus": captured["collectionStatus"],
        "newArticleAnalyses": captured["completed"],
        "failedArticleAnalyses": captured["failed"],
        "freshCaptureCoverage": captured["coverage"],
        "newForecastCommitmentsByEligibility": captured["newForecastCommitments"],
        "commitmentLedgerAudit": captured["forecastCommitmentAudit"],
        "marketPriceReport": {
            "status": price_result["status"],
            "recordedCommitmentHorizons": price_result["committedRows"],
            "priceObservationRows": price_result["priceObservations"],
            "priceSha256": price_result.get("priceSha256"),
            "sourceCommitmentChainSha256": price_result.get(
                "commitmentChainHeadSha256"
            ),
            "statusCounts": price_result.get("statusCounts", {}),
            "horizons": price_result.get("horizons", {}),
        },
        "newMarketPricesFetched": False,
        "freshModelPredictionsOnly": True,
        "eventAwareForecastEnabled": event_aware,
        "historicalAnalysisCohortExcluded": True,
        "firstPublicAvailabilityIndependentlyVerified": False,
        "externalTimestampAuthenticated": False,
        "prospectivePredictionAccuracy": None,
        "trainingEligible": False,
        "tradingEnabled": False,
        "marketCausalityEstablished": False,
        "nextPriceRefresh": (
            "Only after genuinely completed new stock and benchmark closes "
            "exist, run incremental_daily_market_maturity_cli "
            "--include-committed-forecasts with a new immutable output dir."
        ),
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--mappings", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--price-manifest", type=Path)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--fresh-minutes", type=int, default=120)
    parser.add_argument("--skip-collection", action="store_true")
    parser.add_argument("--event-aware", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = cycle(
            args.db, args.mappings, args.prices, args.output,
            model=args.model, limit=args.limit, fresh_minutes=args.fresh_minutes,
            collect=not args.skip_collection, price_manifest=args.price_manifest,
            event_aware=args.event_aware,
        )
        print(json.dumps({
            "status": report["status"],
            "newArticleAnalyses": report.get("newArticleAnalyses", 0),
            "freshCaptureCoverage": report.get("freshCaptureCoverage", {}),
            "newForecastCommitmentsByEligibility": report.get(
                "newForecastCommitmentsByEligibility", {}
            ),
            "marketPriceReport": report.get("marketPriceReport", {}),
            "prospectivePredictionAccuracy": None,
            "output": str(args.output) if report["status"] != "ALREADY_RUNNING" else None,
        }, ensure_ascii=False))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"forward-news-market-cycle: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
