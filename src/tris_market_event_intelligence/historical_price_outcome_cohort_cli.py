"""PIT-safe historical NEWS -> daily stock/benchmark outcome cohort.

Uses already acquired provider-adjusted close CSVs and source-pinned NEWS.
This is an *observational label dataset*, not tradable/backtested forecasts.
As-of inputs are capped at NY 09:30; prior-close to future-close labels are
stored in a separate outcome object, never mixed into predictor features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import date
from itertools import pairwise
from math import isfinite
from pathlib import Path

from tris_market_event_intelligence.historical_shock_research import (
    _context,
    read_daily_prices,
    read_macro_releases,
)
from tris_market_event_intelligence.news_importance_review import _cohort

HORIZONS = (1, 3, 5)
MAX_NEWS_ROWS = 5000


def build_news_price_outcome_cohort(
    *, stock_csv: Path, benchmark_csv: Path, news_cohort: Path,
    ticker: str, benchmark: str, macro_csv: Path | None = None,
) -> dict[str, object]:
    if (not ticker.isascii() or not benchmark.isascii()
            or not ticker.isalnum() or not benchmark.isalnum()
            or not 1 <= len(ticker) <= 12 or not 1 <= len(benchmark) <= 12
            or ticker == benchmark):
        raise ValueError("Expected two distinct bounded ASCII ticker symbols")
    stock, stock_sha = read_daily_prices(stock_csv)
    market, market_sha = read_daily_prices(benchmark_csv)
    raw_news, source_news = _cohort(news_cohort)
    rows_news = source_news["items"]
    if len(rows_news) > MAX_NEWS_ROWS:
        raise ValueError("Excessive news cohort")
    macro, macro_sha = (
        read_macro_releases(macro_csv) if macro_csv is not None else ([], None)
    )
    market_by_day = {bar.day: bar for bar in market}
    # Pairwise stock sessions are used as the observed outcome calendar.
    # Never bridge missing benchmark sessions or missing stock trading dates.
    records: list[dict[str, object]] = []
    news_counts: Counter[str] = Counter()
    for prior, today in pairwise(stock):
        if prior.day not in market_by_day or today.day not in market_by_day:
            continue
        ctx = _context(day=today.day, articles=rows_news, macro=macro)
        cutoff = ctx["asOf"]
        potential_news = ctx["newsAvailableBeforeOpen"]
        # Source association is only a candidate: neither URL nor a matching
        # headline independently verifies that a story drove a stock price.
        news_rows = [{
            "articleId": row["articleId"],
            "source": row["source"],
            "sourceUrl": row["sourceUrl"],
            "textSha256": row["textSha256"],
            "captureSha256": next(
                article["captureSha256"] for article in rows_news
                if article["articleId"] == row["articleId"]
            ),
            "firstObservedAt": row["firstObservedAt"],
            "title": row["title"],
            "issuerAssociationVerified": False,
            "independentlyCausal": False,
        } for row in potential_news]
        if ctx["newsAvailableBeforeOpenTotal"] > len(news_rows):
            news_counts["TRUNCATED_AT_CONTEXT_LIMIT"] += 1
        if not news_rows:
            news_counts["NO_OBSERVED_NEWS_AT_CUTOFF"] += 1
        horizon_labels: dict[str, object] = {}
        stock_by_day = {bar.day: bar for bar in stock}
        aligned_dates = [bar.day for bar in stock if bar.day in market_by_day]
        position = aligned_dates.index(today.day)
        for horizon in HORIZONS:
            future_idx = position + horizon - 1
            if future_idx >= len(aligned_dates):
                continue
            target_day = aligned_dates[future_idx]
            future_stock = stock_by_day[target_day]
            future_market = market_by_day[target_day]
            sr = future_stock.adjusted_close / prior.adjusted_close - 1
            mr = future_market.adjusted_close / market_by_day[prior.day].adjusted_close - 1
            if not isfinite(sr) or not isfinite(mr):
                raise ValueError("Non-finite adjusted return")
            horizon_labels[str(horizon)] = {
                "targetSession": target_day.isoformat(),
                "stockCloseToCloseReturn": sr,
                "benchmarkCloseToCloseReturn": mr,
                "excessReturn": sr - mr,
                "directionVersusBenchmark": (
                    "UP" if sr > mr else "DOWN" if sr < mr else "FLAT"
                ),
            }
        if not horizon_labels:
            continue
        records.append({
            "ticker": ticker,
            "benchmark": benchmark,
            "decisionSession": today.day.isoformat(),
            "asOf": cutoff,
            "features": {
                "lastKnownStockSession": prior.day.isoformat(),
                "lastKnownStockAdjustedClose": prior.adjusted_close,
                "lastKnownBenchmarkAdjustedClose": (
                    market_by_day[prior.day].adjusted_close
                ),
                "news": news_rows,
                "newsCountPresented": len(news_rows),
                "newsCountAvailable": ctx["newsAvailableBeforeOpenTotal"],
                "macroPointInTime": ctx["macroPointInTime"],
                "issuerLinksIndependentlyVerified": False,
            },
            "outcome": {
                "labelAvailability": "AFTER_TARGET_SESSION_CLOSE",
                "horizonsTradingSessions": horizon_labels,
                "notARealTimeTradableEntry": True,
                "closeToCloseContainsPreOpenPriceMovement": True,
            },
            "independentNewsImpactCausalityProven": False,
            "modelPredictionGenerated": False,
            "trainingEligible": False,
        })
    if not records:
        raise ValueError("No common priced outcome sessions")
    with_news = sum(bool(row["features"]["news"]) for row in records)
    without_news = len(records) - with_news
    return {
        "schemaVersion": "historical_news_price_outcome_cohort_v1",
        "status": "OBSERVATIONAL_LABELS_REQUIRE_POINT_IN_TIME_REVIEW",
        "symbol": ticker,
        "benchmarkSymbol": benchmark,
        "stockPriceSha256": stock_sha,
        "benchmarkPriceSha256": market_sha,
        "newsCohortSha256": hashlib.sha256(raw_news).hexdigest(),
        "macroSha256": macro_sha,
        "macroEvidenceProvided": macro_csv is not None,
        "decisionCutoff": "09:30 America/New_York",
        "horizons": list(HORIZONS),
        "observationRows": len(records),
        "rowsWithPreCutoffNews": with_news,
        "rowsWithoutPreCutoffNews": without_news,
        "daysWithTruncatedNews": news_counts["TRUNCATED_AT_CONTEXT_LIMIT"],
        "sourcePublisherTimeVerified": False,
        "sourceIssuerAssociationVerified": False,
        "priceVendorAuthenticityIndependentlyVerified": False,
        "humanGoldCreated": False,
        "trainingReady": False,
        "forwardPredictionsSealed": False,
        "modelPromotionAllowed": False,
        "productionDbModified": False,
        "records": records,
    }


def _private_write(path: Path, data: dict) -> None:
    if path.is_symlink() or path.exists() or not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError("Private output must be new in existing directory")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PIT NEWS features and separate observed EOD stock/benchmark labels"
    )
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--benchmark-symbol", required=True)
    parser.add_argument("--stock-csv", type=Path, required=True)
    parser.add_argument("--benchmark-csv", type=Path, required=True)
    parser.add_argument("--news-cohort", type=Path, required=True)
    parser.add_argument("--macro-csv", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    try:
        if not args.preview and args.output is None:
            raise ValueError("Output path required")
        report = build_news_price_outcome_cohort(
            stock_csv=args.stock_csv,
            benchmark_csv=args.benchmark_csv,
            news_cohort=args.news_cohort,
            macro_csv=args.macro_csv,
            ticker=args.symbol,
            benchmark=args.benchmark_symbol,
        )
        if not args.preview:
            _private_write(args.output, report)
        print(json.dumps({
            "status": report["status"],
            "symbol": report["symbol"],
            "observations": report["observationRows"],
            "newsCoveredSessions": report["rowsWithPreCutoffNews"],
            "newsMissingSessions": report["rowsWithoutPreCutoffNews"],
            "newsTruncatedSessions": report["daysWithTruncatedNews"],
            "trainingReady": False,
            "predictionReady": False,
            "outputCreated": not args.preview,
        }))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        parser.exit(2, "historical-price-outcomes: INVALID_OR_UNSAFE_SOURCE\n")


if __name__ == "__main__":
    main()
