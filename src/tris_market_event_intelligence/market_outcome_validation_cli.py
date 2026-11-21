"""Evaluate saved Gemma article hypotheses against sourced historical daily prices.

Read-only source artifacts; no access to the news production database or live pipeline.
The input price CSV must contain real, licensed/authorized adjusted-close observations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from math import isfinite
from pathlib import Path
from zoneinfo import ZoneInfo

from tris_market_event_intelligence.event_study import DailyClose, PublishedEvent, study_events

DIRECTIONS = {"UP", "DOWN", "NEUTRAL", "UNCERTAIN"}
ZONES = {
    "XKRX": "Asia/Seoul",
    "XNAS": "America/New_York",
    "XNYS": "America/New_York",
}
HORIZONS = (1, 5)
NEUTRAL_THRESHOLD = 0.001


def _load_json(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Expected existing regular JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Expected existing regular CSV: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV missing headers: {sorted(required - set(reader.fieldnames or []))}")
        return list(reader)


def _timestamp(text: str) -> datetime:
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Publication and processing timestamps must be timezone-aware")
    return parsed


def _mappings(path: Path) -> dict[str, dict]:
    rows = _csv(path, {"target", "market", "symbol", "benchmark", "verified_at"})
    result: dict[str, dict] = {}
    for row in rows:
        target = row["target"].strip()
        market = row["market"].strip()
        symbol = row["symbol"].strip()
        benchmark = row["benchmark"].strip()
        verified_at = _timestamp(row["verified_at"].strip())
        if (not target or market not in ZONES or not symbol or not benchmark
                or symbol == benchmark or target in result):
            raise ValueError("Ambiguous or invalid target mapping")
        result[target] = {
            "market": market, "symbol": symbol, "benchmark": benchmark,
            "verifiedAt": verified_at.isoformat(),
        }
    return result


def _prices(path: Path) -> list[DailyClose]:
    rows = _csv(path, {"symbol", "date", "adjusted_close", "source"})
    from datetime import date
    values: list[DailyClose] = []
    for row in rows:
        if not row["source"].strip():
            raise ValueError("Each price observation requires a nonempty source")
        value = float(row["adjusted_close"])
        if not isfinite(value) or value <= 0:
            raise ValueError("Prices must be positive finite adjusted closes")
        values.append(DailyClose(
            symbol=row["symbol"].strip(),
            day=date.fromisoformat(row["date"].strip()),
            adjusted_close=value,
        ))
    return values


def _direction(actual_return: float) -> str:
    if actual_return > NEUTRAL_THRESHOLD:
        return "UP"
    if actual_return < -NEUTRAL_THRESHOLD:
        return "DOWN"
    return "NEUTRAL"


def evaluate(
    articles_dir: Path, mappings_csv: Path, prices_csv: Path, *,
    price_source_description: str,
) -> dict:
    """Generate a descriptive backtest; never represent retrospective runs as live forecasts."""
    if not price_source_description.strip():
        raise ValueError("Price dataset provenance description is required")
    if not articles_dir.is_dir() or articles_dir.is_symlink():
        raise ValueError("Existing nonsymlink Gemma result directory required")
    mappings = _mappings(mappings_csv)
    prices = _prices(prices_csv)
    reports = sorted(articles_dir.glob("*.json"))
    if not reports:
        raise ValueError("No Gemma article JSON files found")
    detail: list[dict] = []
    summaries = Counter()
    seen_articles: set[int] = set()
    for source_file in reports:
        report = _load_json(source_file)
        if report.get("schemaVersion") != "gemma_original_article_review_v2":
            raise ValueError(f"Unexpected Gemma source schema: {source_file}")
        article = report["article"]
        article_id = article["articleId"]
        if type(article_id) is not int or article_id < 1 or article_id in seen_articles:
            raise ValueError("Duplicate or invalid article ID")
        seen_articles.add(article_id)
        published = _timestamp(article["publishedAt"])
        processed = _timestamp(report["processedAt"])
        if processed < published:
            raise ValueError("AI processing predates source publication")
        for ordinal, impact in enumerate(report["analysis"]["impacts"]):
            target = impact["target"].strip()
            predicted = impact["direction"]
            if predicted not in DIRECTIONS or not target:
                raise ValueError("Invalid Gemma impact prediction")
            item = {
                "articleId": article_id, "impactIndex": ordinal,
                "title": article["title"], "target": target,
                "prediction": predicted, "reason": impact["reason"],
                "evidence": impact["evidence_quote"],
                "publishedAt": published.isoformat(),
                "processedAt": processed.isoformat(),
                "model": report["model"], "predictionSourceSha256": _sha(source_file),
                "predictionSealedBeforeOutcome": bool(report.get("predictionSealedBeforeOutcome")),
                "evaluationClass": "PROSPECTIVE_SEALED"
                if report.get("predictionSealedBeforeOutcome") else "RETROSPECTIVE_UNSEALED",
                "horizons": [],
            }
            mapping = mappings.get(target)
            if mapping is None:
                item["status"] = "UNMAPPED"
            elif _timestamp(mapping["verifiedAt"]) > published:
                # Post-event mappings can support retrospective research, but not PIT evidence.
                item["status"] = "MAPPING_NOT_POINT_IN_TIME"
                item["mapping"] = mapping
            else:
                item["mapping"] = mapping
                # event_study uses Seoul dates; convert to exchange-local civil time
                # before passing the localized timestamp to its daily entry policy.
                local_published = published.astimezone(ZoneInfo(ZONES[mapping["market"]]))
                # Daily event_study currently hardcodes Asia/Seoul: compute publication
                # day from local date with a KST proxy to preserve the exchange date.
                kst_date = local_published.date()
                proxy = datetime.combine(kst_date, datetime.min.time(), ZoneInfo("Asia/Seoul"))
                studied = study_events(
                    [PublishedEvent(article_id, mapping["symbol"], proxy)],
                    prices, benchmark_symbol=mapping["benchmark"], horizons=HORIZONS,
                )[0]
                item["entryDay"] = studied.entry_day
                item["status"] = studied.status
                for label in studied.labels:
                    observed = (
                        _direction(label.stock_return)
                        if label.status == "COMPLETED" and label.stock_return is not None else None
                    )
                    verdict = (
                        "ABSTAINED" if predicted == "UNCERTAIN" and observed else
                        "CORRECT" if observed == predicted else
                        "INCORRECT" if observed else "UNAVAILABLE"
                    )
                    item["horizons"].append({
                        "sessions": label.horizon_sessions,
                        "exitDay": label.exit_day,
                        "stockReturn": label.stock_return,
                        "benchmarkReturn": label.benchmark_return,
                        "excessReturn": label.excess_return,
                        "actualDirection": observed,
                        "verdict": verdict,
                        "dataStatus": label.status,
                    })
                    summaries[f"{label.horizon_sessions}d:{verdict}"] += 1
            summaries[item["status"]] += 1
            detail.append(item)
    metrics = {}
    for horizon in HORIZONS:
        correct = summaries[f"{horizon}d:CORRECT"]
        wrong = summaries[f"{horizon}d:INCORRECT"]
        metrics[f"{horizon}d"] = {
            "correct": correct, "incorrect": wrong,
            "abstained": summaries[f"{horizon}d:ABSTAINED"],
            "unavailable": summaries[f"{horizon}d:UNAVAILABLE"],
            "accuracy": correct / (correct + wrong) if correct + wrong else None,
            "denominator": correct + wrong,
        }
    return {
        "schemaVersion": "market_outcome_validation_v1",
        "researchOnly": True,
        "historicalCausalityProven": False,
        "predictionIntegrity": "Per-file SHA256; retrospective unsealed results are not prospective forecasts",
        "entryPolicy": "First daily close on a trading date strictly later than exchange-local publication day",
        "horizons": list(HORIZONS), "neutralThreshold": NEUTRAL_THRESHOLD,
        "priceBasis": "source_provided_adjusted_close",
        "priceSource": price_source_description,
        "pricesSha256": _sha(prices_csv), "mappingsSha256": _sha(mappings_csv),
        "articlesInspected": len(reports), "impactsInspected": len(detail),
        "statusCounts": dict(summaries), "metrics": metrics, "items": detail,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemma-dir", type=Path, required=True)
    parser.add_argument("--mappings", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--price-source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("Output already exists; previous evaluations are immutable")
        result = evaluate(
            args.gemma_dir, args.mappings, args.prices,
            price_source_description=args.price_source,
        )
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        print(json.dumps({
            "status": "COMPLETED", "articles": result["articlesInspected"],
            "impacts": result["impactsInspected"], "metrics": result["metrics"],
            "output": str(args.output),
        }, ensure_ascii=False))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"market-outcome-validation: {exc}\n")


if __name__ == "__main__":
    main()
