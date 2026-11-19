"""Frozen, read-only chronological holdout evaluation; never creates lessons.

Use only after an exploratory run. Reports predictive price co-movement, not
causality; training outcomes are not inserted into prompts or re-used as labels.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import httpx

from tris_market_event_intelligence.event_evidence_policy import (
    article_market,
    content_quality,
    localize_prediction,
)
from tris_market_event_intelligence.event_grounding import (
    enforce_event_grounding,
    event_facts,
)
from tris_market_event_intelligence.llm_runtime import LLMSettings
from tris_market_event_intelligence.news_llm_factory import build_runtime
from tris_market_event_intelligence.sequential_market_learning_cli import (
    evaluate,
    load_articles,
    pre_event_market_context,
    predict_with_repair,
    publication_time,
    timestamp,
    validate_asset_evidence,
)


def evaluate_holdout(
    *, journal: Path, source_db: Path, output: Path, model: object,
    client: httpx.Client, count: int = 100,
) -> dict:
    if not 1 <= count <= 1000:
        raise ValueError("Holdout count must be 1..1000")
    if output.exists() or output.is_symlink():
        raise ValueError("Refusing to overwrite a holdout result")
    with sqlite3.connect(f"file:{journal.resolve()}?mode=ro", uri=True) as db:
        row = db.execute("SELECT settings_json FROM runs WHERE id=1").fetchone()
        if row is None:
            raise ValueError("Missing experimental settings")
        settings = json.loads(row[0])
        bounds = db.execute(
            "SELECT MAX(observed_at), COUNT(*) FROM steps "
            "WHERE phase='COMPLETE'"
        ).fetchone()
        training_ids = {item[0] for item in db.execute(
            "SELECT article_id FROM steps"
        )}
    if not bounds[0]:
        raise ValueError("No finished training articles")
    if settings["sourceDb"] != str(source_db.resolve()):
        raise ValueError("Source archive differs from the training contract")
    cutoff = timestamp(settings["cutoff"])
    cache = settings.get("sourceContentCache")
    kwargs = {"content_cache": Path(cache)} if cache else {}
    holdout, _ = load_articles(
        source_db, 1000, cutoff, partition="holdout", **kwargs,
    )
    last_training = timestamp(bounds[0])
    selected = [
        a for a in holdout if a["articleId"] not in training_ids
        and publication_time(a) > last_training
    ][:count]
    outcomes = []
    from datetime import timedelta

    horizon = timedelta(hours=float(settings["horizonHours"]))
    for article in selected:
        observed = publication_time(article)
        context = {
            "article": {
                "title": article["title"], "summary": article["summary"],
                "publishedAtReported": article["publishedAtReported"],
                "source": article["source"],
                "eventMarket": article_market(article),
                "bodyQuality": content_quality(article),
                "eventFacts": event_facts(article),
            },
            "priorFeedbackLessons": [],
            "marketContext": pre_event_market_context(article, [], client),
        }
        try:
            prediction = enforce_event_grounding(
                localize_prediction(
                    validate_asset_evidence(
                        predict_with_repair(model, context), article,
                    ), article,
                ), article,
            )
            price = evaluate(
                prediction, observed, horizon, settings["interval"], client,
            )
            status = "COMPLETE"
            issue = None
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            prediction = {"impacts": []}
            price = {"impacts": []}
            status = "SKIPPED_INVALID_MODEL"
            issue = str(error)[:300]
        outcomes.append({
            "articleId": article["articleId"], "releaseAt": observed.isoformat(),
            "title": article["title"], "source": article["source"],
            "prediction": prediction, "evaluation": price,
            "status": status, "issue": issue,
        })
    judged = [
        target for item in outcomes
        for target in item["evaluation"]["impacts"]
        if target.get("status") in {"CORRECT", "WRONG"}
    ]
    directional = [
        item for item in judged
        if item.get("direction") in {"POSITIVE", "NEGATIVE"}
    ]
    result = {
        "schemaVersion": "frozen_chronological_holdout_v1",
        "trainingJournal": str(journal), "holdoutSourceDb": str(source_db),
        "pointInTimeCertified": False, "independentSourceVerified": False,
        "learningDisabled": True, "feedbackRulesApplied": 0,
        "lastTrainingPublishedAt": last_training.isoformat(),
        "holdoutSelected": len(selected),
        "holdoutCompleted": sum(x["status"] == "COMPLETE" for x in outcomes),
        "targetsPriceScored": len(judged),
        "directionalPriceScored": len(directional),
        "directionalPriceMatches": sum(
            x["status"] == "CORRECT" for x in directional
        ),
        "causalAttributionEstablished": False,
        "warning": ("Retrospective content is not PIT-certified. Model prompting "
                    "uses no learned feedback. This evaluates a frozen baseline, "
                    "not an independently verified gain from training."),
        "articles": outcomes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return {k: v for k, v in result.items() if k != "articles"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    if args.journal.is_symlink() or args.source_db.is_symlink():
        parser.error("Symlink inputs are not allowed")
    with sqlite3.connect(f"file:{args.journal.resolve()}?mode=ro", uri=True) as db:
        row = db.execute("SELECT settings_json FROM runs WHERE id=1").fetchone()
    if row is None:
        parser.error("Missing study settings")
    settings = json.loads(row[0])
    model = build_runtime(
        replace(
            LLMSettings.from_environment(explicit=True),
            model_id=settings["model"],
        ), confirm_model=True,
    )
    with httpx.Client(timeout=20, follow_redirects=False) as client:
        print(json.dumps(evaluate_holdout(
            journal=args.journal, source_db=args.source_db,
            output=args.output, model=model, client=client,
            count=args.count,
        ), ensure_ascii=False))


if __name__ == "__main__":
    main()
