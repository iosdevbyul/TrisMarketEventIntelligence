"""Join real automatic Gemma SQLite ledger to actual stored adjusted daily closes.

This is a *read-only retrospective* research bridge, not a live forecast seal.
It does not turn historical model runs into pre-outcome predictions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from tris_market_event_intelligence.gemma_original_article_batch_cli import MODEL_PATTERN
from tris_market_event_intelligence.market_outcome_validation_cli import evaluate
from tris_market_event_intelligence.news_content_quality import media_page
from tris_market_event_intelligence.operational_news_pit_readiness_cli import (
    inspect_live_news,
)

MAX_AUTOMATIC_ARTICLES = 3000


def _automatically_analyzed(
    news_db: Path, model: str, *, limit: int,
) -> tuple[list[dict], dict]:
    if not MODEL_PATTERN.fullmatch(model) or not 1 <= limit <= MAX_AUTOMATIC_ARTICLES:
        raise ValueError("Invalid model name or bounded article limit")
    if not news_db.is_file() or news_db.is_symlink():
        raise ValueError("Expected real local news.sqlite3 source")
    audit = inspect_live_news(news_db, limit=MAX_AUTOMATIC_ARTICLES)
    if audit["truncated"]:
        raise ValueError("News DB audit truncated; refusing partial source coverage")
    status = {row["articleId"]: row for row in audit["rows"]}
    uri = news_db.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=15)) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=ON")
        exists = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='automatic_news_analyses'"
        ).fetchone()
        if exists is None:
            raise ValueError("Automatic Gemma analysis ledger is absent")
        rows = db.execute(
            """SELECT a.id, a.source, a.title, a.url, a.published_at,
                      a.detected_at, f.body, f.body_sha256, f.status AS text_status,
                      n.model, n.processed_at, n.analysis_json
               FROM automatic_news_analyses n
               JOIN articles a ON a.id=n.article_id
               JOIN article_full_text f ON f.article_id=a.id
               WHERE n.model=? AND n.status='COMPLETED'
                 AND n.body_sha256=f.body_sha256
               ORDER BY a.id, n.processed_at""", (model,),
        ).fetchall()
        # One row per current article/model/body identity; completed ledger
        # is immutable, but fail closed on corrupt duplicate completions.
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate completed model rows for article")
        selected = rows[:limit]
        reports = []
        invalid = []
        dual_excluded = []
        media_excluded = []
        for row in selected:
            article_id = row["id"]
            if media_page(row["url"]):
                media_excluded.append(article_id)
                continue
            source = status.get(article_id)
            body = row["body"]
            if (source is None or not source["sourceIntegrityReviewed"]
                    or row["text_status"] != "COMPLETE"
                    or not isinstance(body, str)
                    or hashlib.sha256(body.encode()).hexdigest() != row["body_sha256"]):
                invalid.append(article_id)
                continue
            try:
                analysis = json.loads(row["analysis_json"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Stored Gemma JSON invalid for article {article_id}"
                ) from exc
            if (not isinstance(analysis, dict)
                    or not isinstance(analysis.get("impacts"), list)):
                raise TypeError(f"Stored Gemma impact schema invalid: {article_id}")
            if analysis.get("forecastVersion") == "event_aware_dual_v1":
                dual_excluded.append(article_id)
                continue  # Scored only by the horizon-aware commitment evaluator.
            for impact in analysis["impacts"]:
                if (not isinstance(impact, dict)
                        or not isinstance(impact.get("target"), str)
                        or not isinstance(impact.get("direction"), str)
                        or not isinstance(impact.get("reason"), str)
                        or not isinstance(impact.get("evidence_quote"), str)):
                    raise TypeError(f"Stored impact lacks source evidence: {article_id}")
            reports.append({
                "schemaVersion": "gemma_original_article_review_v2",
                "model": row["model"], "processedAt": row["processed_at"],
                "article": {
                    "articleId": article_id, "source": row["source"],
                    "title": row["title"], "url": row["url"],
                    "publishedAt": row["published_at"],
                    "detectedAt": row["detected_at"],
                    "bodySha256": row["body_sha256"],
                },
                "analysis": analysis,
                "priceReturnsEvaluated": False,
                "predictionSealedBeforeOutcome": False,
                "researchOnly": True,
                "automaticLedgerOrigin": True,
                "forecastTimestampAuthenticatedAsPreOutcome": False,
            })
    return reports, {
        "completedCurrentBodyModelRows": len(rows),
        "selectedCompletedRows": len(selected),
        "excludedCorruptSourceIds": invalid,
        "excludedDualForecastArticleIds": dual_excluded,
        "excludedAudioVideoArticleIds": media_excluded,
        "sourceArticleCount": audit["totalArticlesAtScan"],
        "sourceBlockerCounts": audit["blockerCounts"],
        "truncated": len(rows) > limit,
    }


def evaluate_automatic_outcomes(
    news_db: Path, mappings: Path, prices: Path, *,
    model: str, limit: int = MAX_AUTOMATIC_ARTICLES,
    price_source: str, price_manifest: Path | None = None,
) -> dict:
    """Compare recorded original Gemma impacts with real daily market data."""
    reports, coverage = _automatically_analyzed(news_db, model, limit=limit)
    if not reports:
        raise ValueError("No source-admissible completed automatic Gemma articles")
    with tempfile.TemporaryDirectory(prefix="tmie-automatic-outcomes-") as temp:
        directory = Path(temp)
        for report in reports:
            path = directory / f"{report['article']['articleId']}.json"
            path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        result = evaluate(
            directory, mappings, prices,
            price_source_description=price_source,
            news_db=news_db,
            price_manifest=price_manifest,
        )
    result.update({
        "schemaVersion": "automatic_gemma_daily_outcomes_v1",
        "sourceType": "REAL_AUTOMATIC_SQLITE_GEMMA_LEDGER",
        "sourceModel": model,
        "automaticLedgerCoverage": coverage,
        "historicalEvaluationOnly": True,
        "marketPricesAnalyzed": True,
        "sourcedAdjustedDailyPriceFileCompared": True,
        "actualMarketOutcomeProviderIndependentlyVerified": False,
        "forecastSealedBeforeOutcomes": False,
        "prospectivePredictionAccuracy": None,
        "trainingEligible": False,
        "productionDatabaseModified": False,
        "explanation": (
            "Stored original-body Gemma analysis is joined to actual "
            "source-provided adjusted daily close files and reviewed "
            "instrument mappings. Evaluated after the fact; no verified "
            "prospective forecast, market cause or time-of-first-publication."
        ),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real automatic Gemma SQLite ledger → adjusted daily price research",
    )
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--model", default="gemma4:e2b")
    parser.add_argument("--limit", type=int, default=MAX_AUTOMATIC_ARTICLES)
    parser.add_argument("--mappings", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--price-manifest", type=Path)
    parser.add_argument("--price-source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("Output already exists; original evidence immutable")
        result = evaluate_automatic_outcomes(
            args.db, args.mappings, args.prices, model=args.model,
            limit=args.limit, price_source=args.price_source,
            price_manifest=args.price_manifest,
        )
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(json.dumps({
            "status": "COMPLETED",
            "storedGemmaArticles": result["articlesInspected"],
            "impacts": result["impactsInspected"],
            "statusCounts": {
                k: v for k, v in result["statusCounts"].items()
                if ":" not in k
            },
            "dailyMetrics": result["metrics"],
            "marketPricesAnalyzed": True,
            "prospectivePredictionAccuracy": None,
            "output": str(args.output),
        }, ensure_ascii=False))
    except (ValueError, TypeError, KeyError, OSError, sqlite3.Error) as exc:
        parser.exit(2, f"automatic-gemma-market-outcomes: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
