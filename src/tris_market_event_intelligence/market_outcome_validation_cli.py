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

from tris_market_event_intelligence.operational_news_pit_readiness_cli import inspect_live_news
from tris_market_event_intelligence.operational_news_price_bridge_cli import (
    ADMISSION_BLOCKERS, _stored_articles,
)
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
        kind = (row.get("mapping_kind") or "DIRECT_EQUITY").strip()
        limitation = (row.get("proxy_limitation") or "").strip()
        evidence_url = (row.get("evidence_url") or "").strip()
        if kind not in {"DIRECT_EQUITY", "SECTOR_ETF_PROXY"}:
            raise ValueError("Unsupported mapping kind")
        if kind == "SECTOR_ETF_PROXY" and (not limitation or not evidence_url):
            raise ValueError("Sector ETF proxy requires limitation and source evidence")
        result[target] = {
            "market": market, "symbol": symbol, "benchmark": benchmark,
            "verifiedAt": verified_at.isoformat(),
            "mappingKind": kind, "proxyLimitation": limitation,
            "evidenceUrl": evidence_url,
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
    news_db: Path | None = None,
    price_manifest: Path | None = None,
) -> dict:
    """Generate a descriptive backtest; never represent retrospective runs as live forecasts."""
    if not price_source_description.strip():
        raise ValueError("Price dataset provenance description is required")
    if not articles_dir.is_dir() or articles_dir.is_symlink():
        raise ValueError("Existing nonsymlink Gemma result directory required")
    mappings = _mappings(mappings_csv)
    prices = _prices(prices_csv)
    price_manifest_verified = False
    if price_manifest is not None:
        provenance = _load_json(price_manifest)
        if (provenance.get("outputSha256") != _sha(prices_csv)
                or provenance.get("mappingsSha256") != _sha(mappings_csv)):
            raise ValueError("Price recovery manifest SHA mismatch")
        price_manifest_verified = True
    source_audit = inspect_live_news(news_db, limit=3000) if news_db else None
    if source_audit and source_audit["truncated"]:
        raise ValueError("Operational news source audit truncated")
    source_rows = _stored_articles(news_db) if news_db else {}
    audit_rows = (
        {row["articleId"]: row for row in source_audit["rows"]}
        if source_audit else {}
    )
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
        if report.get("predictionSealedBeforeOutcome") is not False:
            raise ValueError("Original Gemma batch must be explicitly retrospective-unsealed")
        article = report["article"]
        article_id = article["articleId"]
        if type(article_id) is not int or article_id < 1 or article_id in seen_articles:
            raise ValueError("Duplicate or invalid article ID")
        seen_articles.add(article_id)
        published = _timestamp(article["publishedAt"])
        processed = _timestamp(report["processedAt"])
        if processed < published:
            raise ValueError("AI processing predates source publication")
        source_gate = "SOURCE_DB_NOT_REQUESTED"
        if source_audit is not None:
            audit_item = audit_rows.get(article_id)
            observed = source_rows.get(article_id)
            if audit_item is None or observed is None:
                source_gate = "NEWS_ARTICLE_ABSENT"
            elif ADMISSION_BLOCKERS.intersection(audit_item["blockers"]):
                source_gate = "NEWS_ORIGINAL_INCOMPLETE_OR_DUPLICATE"
            elif (not isinstance(article.get("url"), str)
                  or article["url"] != observed["url"]
                  or not isinstance(article.get("bodySha256"), str)
                  or article["bodySha256"] != observed["body_sha256"]
                  or observed["status"] != "COMPLETE"
                  or not isinstance(article.get("detectedAt"), str)
                  or _timestamp(article["detectedAt"]) != _timestamp(observed["detected_at"])
                  or _timestamp(article["publishedAt"]) != _timestamp(observed["published_at"])):
                source_gate = "NEWS_SOURCE_IDENTITY_MISMATCH"
            else:
                source_gate = "STORED_ORIGINAL_SHA_MATCHED"
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
                "evaluationClass": "RETROSPECTIVE_UNSEALED",
                "sourceAuditStatus": source_gate,
                "storyCandidateId": (audit_rows[article_id]["storyCandidateId"]
                                     if article_id in audit_rows else None),
                "horizons": [],
            }
            mapping = mappings.get(target)
            if source_audit is not None and source_gate != "STORED_ORIGINAL_SHA_MATCHED":
                item["status"] = "NEWS_SOURCE_NOT_ADMITTED"
                for horizon in HORIZONS:
                    summaries[f"{horizon}d:UNAVAILABLE"] += 1
                    summaries[f"{horizon}d:relative:UNAVAILABLE"] += 1
            elif mapping is None:
                item["status"] = "UNMAPPED"
                for horizon in HORIZONS:
                    summaries[f"{horizon}d:UNAVAILABLE"] += 1
                    summaries[f"{horizon}d:relative:UNAVAILABLE"] += 1
            else:
                mapping_is_pit = _timestamp(mapping["verifiedAt"]) <= published
                item["mapping"] = mapping
                item["mappingPointInTime"] = mapping_is_pit
                if not mapping_is_pit:
                    # A documented retrospective linkage can be studied but never
                    # promoted to a prediction available at publication time.
                    item["evaluationClass"] = "RETROSPECTIVE_POST_EVENT_MAPPING"
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
                    relative_direction = (
                        _direction(label.excess_return)
                        if label.status == "COMPLETED" and label.excess_return is not None else None
                    )
                    relative_verdict = (
                        "ABSTAINED" if predicted == "UNCERTAIN" and relative_direction else
                        "CORRECT" if relative_direction == predicted else
                        "INCORRECT" if relative_direction else "UNAVAILABLE"
                    )
                    item["horizons"].append({
                        "sessions": label.horizon_sessions,
                        "exitDay": label.exit_day,
                        "stockReturn": label.stock_return,
                        "benchmarkReturn": label.benchmark_return,
                        "excessReturn": label.excess_return,
                        "actualDirection": observed,
                        "verdict": verdict,
                        "relativeActualDirection": relative_direction,
                        "relativeVerdict": relative_verdict,
                        "dataStatus": label.status,
                    })
                    summaries[f"{label.horizon_sessions}d:{verdict}"] += 1
                    summaries[f"{label.horizon_sessions}d:relative:{relative_verdict}"] += 1
            summaries[item["status"]] += 1
            detail.append(item)
    metrics = {}
    baselines = {}
    for horizon in HORIZONS:
        correct = summaries[f"{horizon}d:CORRECT"]
        wrong = summaries[f"{horizon}d:INCORRECT"]
        relative_correct = summaries[f"{horizon}d:relative:CORRECT"]
        relative_wrong = summaries[f"{horizon}d:relative:INCORRECT"]
        comparable = [
            row["horizons"][HORIZONS.index(horizon)]["actualDirection"]
            for row in detail
            if len(row["horizons"]) == len(HORIZONS)
            and row["horizons"][HORIZONS.index(horizon)]["verdict"]
            in {"CORRECT", "INCORRECT"}
        ]
        comparable_rows = [
            row for row in detail
            if len(row["horizons"]) == len(HORIZONS)
            and row["horizons"][HORIZONS.index(horizon)]["verdict"]
            in {"CORRECT", "INCORRECT"}
        ]
        baselines[f"{horizon}d"] = {
            "denominator": len(comparable),
            "alwaysUp": sum(direction == "UP" for direction in comparable),
            "alwaysDown": sum(direction == "DOWN" for direction in comparable),
            "alwaysNeutral": sum(direction == "NEUTRAL" for direction in comparable),
            "meaning": "Descriptive constant-class agreement on the same scored cohort",
        }
        metrics[f"{horizon}d"] = {
            "correct": correct,
            "distinctArticlesScored": len({r["articleId"] for r in comparable_rows}),
            "distinctPublicationDatesScored": len({
                r["publishedAt"][:10] for r in comparable_rows
            }),
            "distinctStoryCandidateGroupsScored": len({
                r["storyCandidateId"] for r in comparable_rows
                if r["storyCandidateId"] is not None
            }), "incorrect": wrong,
            "abstained": summaries[f"{horizon}d:ABSTAINED"],
            "unavailable": summaries[f"{horizon}d:UNAVAILABLE"],
            "accuracy": correct / (correct + wrong) if correct + wrong else None,
            "denominator": correct + wrong,
            "totalImpacts": len(detail),
            "coverage": (correct + wrong) / len(detail) if detail else None,
            "relative": {
                "correct": relative_correct,
                "incorrect": relative_wrong,
                "abstained": summaries[f"{horizon}d:relative:ABSTAINED"],
                "unavailable": summaries[f"{horizon}d:relative:UNAVAILABLE"],
                "accuracy": (
                    relative_correct / (relative_correct + relative_wrong)
                    if relative_correct + relative_wrong else None
                ),
                "denominator": relative_correct + relative_wrong,
            },
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
        "priceRecoveryManifestVerified": price_manifest_verified,
        "priceRecoveryManifestSha256": (
            _sha(price_manifest) if price_manifest is not None else None
        ),
        "articlesInspected": len(reports), "impactsInspected": len(detail),
        "statusCounts": dict(summaries), "metrics": metrics, "items": detail,
        "constantDirectionBaselines": baselines,
        "marketReturnObservedFromSourcedDailyClose": True,
        "prospectivePredictionAccuracy": None,
        "predictionSealAuthenticated": False,
        "sourceOriginIndependentlyAuthenticated": False,
        "priceProviderIndependentlyAuthenticated": False,
        "trainingEligible": False,
        "operationalNewsDatabaseAudited": news_db is not None,
        "operationalNewsAudit": ({
            "sourceArticles": source_audit["totalArticlesAtScan"],
            "articlesWithoutListedBlockers": source_audit["articlesWithoutListedBlockers"],
            "completeOriginalBodies": source_audit["contentCompleteCount"],
            "databaseModified": False,
        } if source_audit else None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemma-dir", type=Path, required=True)
    parser.add_argument("--mappings", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--price-source", required=True)
    parser.add_argument("--price-manifest", type=Path,
                        help="Optional SHA-pinned recovery manifest for the exact price CSV")
    parser.add_argument("--news-db", type=Path,
                        help="Optional real operational news.sqlite3 source identity check")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("Output already exists; previous evaluations are immutable")
        result = evaluate(
            args.gemma_dir, args.mappings, args.prices,
            price_source_description=args.price_source, news_db=args.news_db,
            price_manifest=args.price_manifest,
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
