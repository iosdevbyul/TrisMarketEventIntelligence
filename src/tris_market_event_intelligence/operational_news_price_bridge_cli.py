"""Link *actual stored news* to sourced market-minute outcomes (read only).

This is an operator-supplied, explicitly mapped descriptive research bridge.
It does not generate issuer matches, market data, or sealed model forecasts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import datetime
from pathlib import Path

from tris_market_event_intelligence.operational_news_pit_readiness_cli import (
    inspect_live_news,
)
from tris_market_event_intelligence.point_in_time_source_validation_cli import (
    EVENT_COLUMNS,
    _time,
    evaluate_sources,
    load_forecasts,
    rows,
)

ADMISSION_BLOCKERS = frozenset({
    "ORIGINAL_BODY_INCOMPLETE",
    "BODY_INTEGRITY_OR_PDF_SPARSITY",
    "ORIGINAL_CAPTURE_INTEGRITY_FAILED",
    "FEED_CAPTURE_INTEGRITY_FAILED",
    "INVALID_ARTICLE_URL",
    "EXACT_URL_ALREADY_PRESENT",
    "STORY_GROUP_HAS_EARLIER_ARTICLE",
    "STORY_GROUP_UNAVAILABLE",
    "COLLECTION_CLOCK_INVALID",
    "PUBLISHER_TIME_MISSING_OR_INVALID",
    "PUBLISHER_TIME_AFTER_DETECTION",
    "DETECTION_PRECEDES_FEED_CAPTURE",
    "BODY_CAPTURE_TIME_INVALID",
})


def _stored_articles(path: Path) -> dict[int, sqlite3.Row]:
    uri = path.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=10)) as db:
        db.row_factory = sqlite3.Row
        return {
            row["id"]: row for row in db.execute(
                "SELECT a.id, a.url, a.published_at, a.detected_at, "
                "f.status, f.body, f.body_sha256, f.observed_at "
                "FROM articles a LEFT JOIN article_full_text f ON f.article_id=a.id"
            )
        }


def evaluate_operational_news_prices(
    news_db: Path,
    mappings_csv: Path,
    source_manifest: Path,
    sessions_csv: Path,
    *,
    evidence_csv: Path | None = None,
    halts_csv: Path | None = None,
) -> dict[str, object]:
    """Show per-article source admission and actual supplied market returns.

    A body SHA is an identity guard, not proof of original information
    availability. All derived outcomes remain unverified descriptive research.
    """
    audit = inspect_live_news(news_db, limit=3000)
    if audit["truncated"]:
        raise ValueError("News source truncated; cannot audit all mapped articles")
    stored = _stored_articles(news_db)
    checked = {row["articleId"]: row for row in audit["rows"]}
    raw = rows(mappings_csv, EVENT_COLUMNS | {"body_sha256"}, 2000)
    forecasts = load_forecasts(mappings_csv)
    if len(raw) != len(forecasts):
        raise ValueError("Prediction row count mismatch")
    admissions: dict[str, dict[str, object]] = {}
    allowed: set[str] = set()
    seen: set[tuple[int, str, str]] = set()
    for mapping, (_, forecast) in zip(raw, forecasts, strict=True):
        article = forecast.event
        prediction_id = forecast.prediction_id
        key = (article.article_id, forecast.target, forecast.horizon_name)
        if key in seen:
            raise ValueError("Duplicate article/target/horizon mapping")
        seen.add(key)
        db_row = stored.get(article.article_id)
        audit_row = checked.get(article.article_id)
        blocked = []
        if db_row is None or audit_row is None:
            blocked.append("ARTICLE_NOT_IN_OPERATIONAL_DB")
        else:
            blocked.extend(
                sorted(ADMISSION_BLOCKERS.intersection(audit_row["blockers"]))
            )
            if (db_row["url"] != article.source_url
                    or _time(db_row["detected_at"]) != _time(mapping["detected_at"])
                    or (db_row["published_at"] is None or
                        _time(db_row["published_at"]) != _time(mapping["published_at"]))):
                blocked.append("NEWS_IDENTITY_OR_CLOCK_MISMATCH")
            digest = mapping["body_sha256"]
            body = db_row["body"]
            if (db_row["status"] != "COMPLETE" or not isinstance(body, str)
                    or not isinstance(digest, str) or len(digest) != 64
                    or hashlib.sha256(body.encode()).hexdigest() != digest
                    or db_row["body_sha256"] != digest):
                blocked.append("ORIGINAL_BODY_DIGEST_MISMATCH")
            captured = db_row["observed_at"]
            if (forecast.mode == "LIVE" and isinstance(captured, str)
                    and _time(captured) > forecast.forecast_at):
                blocked.append("ORIGINAL_BODY_CAPTURED_AFTER_LIVE_SEAL")
        if not blocked:
            allowed.add(prediction_id)
        admissions[prediction_id] = {
            "articleId": article.article_id,
            "predictionId": prediction_id,
            "target": forecast.target,
            "horizon": forecast.horizon_name,
            "admission": "ADMITTED_FOR_DESCRIPTIVE_RESEARCH" if not blocked
                         else "NEWS_SOURCE_BLOCKED",
            "blockers": sorted(set(blocked)),
        }

    result = evaluate_sources(
        mappings_csv, source_manifest, sessions_csv,
        evidence_csv=evidence_csv, halts_csv=halts_csv,
        admitted_prediction_ids=allowed,
    )
    for item in result["items"]:
        gate = admissions[item["predictionId"]]
        item["operationalSource"] = gate
        item["marketOutcomeCausallyCertified"] = False
        item["sealedProspectiveForecastCertified"] = False
        if gate["blockers"]:
            # Fail closed even if existing source validator changes behavior.
            item.pop("observation", None)
            item["status"] = "OPERATIONAL_ARTICLE_NOT_ADMITTED"
    result["statusCounts"] = dict(sorted(Counter(
        item["status"] for item in result["items"]
    ).items()))
    result.update({
        "schemaVersion": "operational_news_price_bridge_v1",
        "newsTotalArticles": audit["totalArticlesAtScan"],
        "newsWithoutAuditBlockers": audit["articlesWithoutListedBlockers"],
        "newsOriginalComplete": audit["contentCompleteCount"],
        "mappingRows": len(raw),
        "operationalSourceAdmitted": len(allowed),
        "operationalSourceBlocked": len(raw) - len(allowed),
        "productionDatabaseModified": False,
        "independentlyVerifiedMarketOutcome": False,
        "independentlyVerifiedArticleRelease": False,
        "trainingEligible": False,
        "notes": (
            "Supplied forecast, first-public time, mapping and minute quotes "
            "are not independently source authenticated. Recorded prices are "
            "descriptive, not demonstrated causes or sealed live performance."
        ),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only live-news to market-price bridge")
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--mappings", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--sessions", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--halts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = evaluate_operational_news_prices(
            args.db, args.mappings, args.source_manifest, args.sessions,
            evidence_csv=args.evidence, halts_csv=args.halts,
        )
        if args.output.is_symlink():
            raise ValueError("Output symlink not permitted")
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(json.dumps({
            "mappingRows": report["mappingRows"],
            "admitted": report["operationalSourceAdmitted"],
            "blocked": report["operationalSourceBlocked"],
            "marketStatusCounts": report["statusCounts"],
            "output": str(args.output),
            "productionDatabaseModified": False,
        }, ensure_ascii=False))
    except (ValueError, KeyError, TypeError, sqlite3.Error, OSError) as exc:
        parser.exit(2, f"news-market-price-bridge: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
