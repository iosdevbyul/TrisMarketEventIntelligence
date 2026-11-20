"""Compare frozen sector-direction hypotheses to observed ETF and benchmark returns.

No hidden market-data download, issuer inference, model rerun or PIT certification.
Price rows are retrospective research observations, not attribution or a trading backtest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from tris_market_event_intelligence.historical_shock_research import read_daily_prices

HORIZONS = (1, 3, 5)
MARKET_CLOCKS = {
    "XKRX": ("Asia/Seoul", time(9, 0)),
    "XNAS": ("America/New_York", time(9, 30)),
    "XNYS": ("America/New_York", time(9, 30)),
}


def read_manifest(path: Path) -> tuple[dict, dict]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("PRICE_MANIFEST_NOT_FOUND")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("schemaVersion") != "sector_price_manifest_v1":
        raise ValueError("PRICE_MANIFEST_SCHEMA_INVALID")
    if obj.get("market") not in MARKET_CLOCKS:
        raise ValueError("PRICE_MANIFEST_MARKET_INVALID")
    benchmark = obj.get("benchmark")
    sectors = obj.get("sectors")
    if not isinstance(benchmark, dict) or not isinstance(sectors, dict):
        raise ValueError("PRICE_MANIFEST_MAPPING_INVALID")
    if len(sectors) > 20 or any(not isinstance(x, dict) for x in sectors.values()):
        raise ValueError("PRICE_MANIFEST_MAPPING_INVALID")
    entries = {"benchmark": benchmark, **sectors}
    series = {}
    for key, item in entries.items():
        symbol, name = item.get("symbol"), item.get("csv")
        if (not isinstance(symbol, str) or not 1 <= len(symbol) <= 24
                or not isinstance(name, str) or name != Path(name).name
                or name in {"", ".", ".."} or "/" in name or "\\" in name):
            raise ValueError("PRICE_MANIFEST_SYMBOL_OR_PATH_INVALID")
        bars, digest = read_daily_prices(path.parent / name)
        series[key] = {
            "symbol": symbol, "bars": bars, "sha256": digest,
            "positions": {row.day: i for i, row in enumerate(bars)},
        }
    return obj, series


def evaluate_sector_prices(analysis: dict, manifest: dict, series: dict) -> dict:
    if (analysis.get("schemaVersion") != "direct_news_sector_impact_v1"
            or analysis.get("priceDataAccessed") is not False
            or analysis.get("originalPredictionsModified") is not False):
        raise ValueError("DIRECT_ANALYSIS_CONTRACT_INVALID")
    if not isinstance(analysis.get("articles"), list):
        raise ValueError("DIRECT_ANALYSIS_ARTICLES_INVALID")
    market = manifest["market"]
    tz_name, opening = MARKET_CLOCKS[market]
    zone = ZoneInfo(tz_name)
    reference = series["benchmark"]
    records = []
    counts = Counter()
    matched_by_horizon = Counter()
    total_by_horizon = Counter()
    for article in analysis["articles"]:
        article_id = article["articleId"]
        published = article.get("publishedAtReported")
        released = None
        try:
            released = datetime.fromisoformat(published)
            if released.utcoffset() is None:
                released = None
        except (ValueError, TypeError):
            pass
        for row in article.get("sectors", []):
            sector, prediction = row["sector"], row["direction"]
            item = {
                "articleId": article_id, "sector": sector,
                "predictedDirection": prediction,
                "predictionSourceQuote": row.get("sourceQuote"),
                "publishedAtReported": published,
                "publishedAtIndependentlyVerified": False,
                "causalAttributionVerified": False,
                "outcomes": {},
            }
            target = series.get(sector)
            if prediction not in {"POSITIVE", "NEGATIVE"}:
                item["status"] = "NON_DIRECTIONAL_HYPOTHESIS"
            elif released is None:
                item["status"] = "PUBLISHER_CLOCK_NOT_PROVIDED"
            elif target is None:
                item["status"] = "SECTOR_PRICE_SERIES_NOT_CONFIGURED"
            else:
                common = sorted(set(reference["positions"]) & set(target["positions"]))
                # The first market OPEN strictly after publication avoids treating
                # an intraday release as information already known at today's open.
                first = next((
                    i for i, d in enumerate(common)
                    if datetime.combine(d, opening, tzinfo=zone).astimezone(UTC)
                    > released.astimezone(UTC)
                ), None)
                if first is None or first == 0:
                    item["status"] = "NO_POST_RELEASE_COMMON_SESSION"
                else:
                    before = common[first - 1]
                    target_prior = target["positions"][before]
                    reference_prior = reference["positions"][before]
                    for horizon in HORIZONS:
                        finish = first + horizon - 1
                        if finish >= len(common):
                            continue
                        # Refuse to bridge even one missing session in either
                        # instrument's underlying adjusted daily series.
                        dates = common[first - 1:finish + 1]
                        if any(
                            target["positions"][b] != target["positions"][a] + 1
                            or reference["positions"][b] != reference["positions"][a] + 1
                            for a, b in zip(dates, dates[1:])
                        ):
                            continue
                        t = target["bars"][target["positions"][common[finish]]]
                        b = reference["bars"][reference["positions"][common[finish]]]
                        sector_return = t.adjusted_close / target["bars"][target_prior].adjusted_close - 1
                        reference_return = b.adjusted_close / reference["bars"][reference_prior].adjusted_close - 1
                        excess = sector_return - reference_return
                        observed = "POSITIVE" if excess > 0 else "NEGATIVE" if excess < 0 else "NEUTRAL"
                        consistent = observed == prediction
                        item["outcomes"][str(horizon)] = {
                            "baselineCloseSession": before.isoformat(),
                            "outcomeCloseSession": common[finish].isoformat(),
                            "sectorReturn": sector_return,
                            "benchmarkReturn": reference_return,
                            "excessReturn": excess,
                            "observedRelativeDirection": observed,
                            "directionMatched": consistent,
                            "returnWindow": "PREVIOUS_CLOSE_TO_FUTURE_CLOSE",
                        }
                        total_by_horizon[horizon] += 1
                        matched_by_horizon[horizon] += consistent
                    item["status"] = (
                        "RETROSPECTIVE_RETURNS_COMPUTED"
                        if item["outcomes"] else "INSUFFICIENT_CONTIGUOUS_PRICE_SESSIONS"
                    )
                if item["status"] == "RETROSPECTIVE_RETURNS_COMPUTED":
                    item["targetSymbol"] = target["symbol"]
                    item["benchmarkSymbol"] = reference["symbol"]
            counts[item["status"]] += 1
            records.append(item)
    return {
        "schemaVersion": "direct_sector_price_outcomes_v1",
        "originalHoldoutSha256": analysis["originalHoldoutSha256"],
        "analysisArticleCount": len(analysis["articles"]),
        "market": market, "benchmarkSymbol": reference["symbol"],
        "priceSeriesDigests": {key: value["sha256"] for key, value in sorted(series.items())},
        "statusCounts": dict(sorted(counts.items())),
        "horizonComparisonCounts": {
            str(h): {"observed": total_by_horizon[h], "matched": matched_by_horizon[h]}
            for h in HORIZONS
        },
        "priceDataAccessed": True,
        "retrospectiveEvaluationOnly": True,
        "originalPredictionSealedBeforePricesIndependentlyVerified": False,
        "pointInTimeCertified": False,
        "causalImpactVerified": False,
        "modelTrained": False, "originalPredictionsModified": False,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--price-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.analysis.is_symlink() or not args.analysis.is_file():
        parser.error("Analysis report not found")
    if args.output.exists() or args.output.is_symlink():
        parser.error("Refusing to overwrite existing outcome report")
    original = args.analysis.read_bytes()
    manifest, sources = read_manifest(args.price_manifest)
    report = evaluate_sector_prices(json.loads(original), manifest, sources)
    report["analysisReportSha256"] = hashlib.sha256(original).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    handle = os.open(args.output, flags, 0o600)
    with os.fdopen(handle, "w", encoding="utf-8") as out:
        json.dump(report, out, ensure_ascii=False, indent=2)
    print(json.dumps({
        key: value for key, value in report.items()
        if key not in {"records", "priceSeriesDigests"}
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
