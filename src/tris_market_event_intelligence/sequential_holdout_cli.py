"""Frozen, read-only chronological holdout evaluation; never creates lessons.

Use only after an exploratory run. Reports predictive price co-movement, not
causality; training outcomes are not inserted into prompts or re-used as labels.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
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


def price_direction_comparison(cases: list[dict]) -> dict:
    """Paired controls on exactly the same observed targets.

    These are co-movement labels, not causal impact ground truth. The model
    must beat trivial directional constants on held-out *priced* targets before
    anyone may report useful signal. Empty/unsupported targets are tracked as
    coverage failures rather than treated as correct neutral predictions.
    """
    outcomes = [
        row for case in cases
        for row in case["evaluation"].get("impacts", [])
        if row.get("status") in {"CORRECT", "WRONG"}
        and row.get("actualDirection") in {"POSITIVE", "NEGATIVE", "NEUTRAL"}
    ]
    actual = Counter(row["actualDirection"] for row in outcomes)
    predicted = Counter(row.get("direction") for row in outcomes)
    directional = [
        row for row in outcomes if row.get("direction") in {"POSITIVE", "NEGATIVE"}
    ]
    paired = len(outcomes)
    correct = sum(row["status"] == "CORRECT" for row in outcomes)
    baselines = {
        direction: {
            "correct": actual[direction],
            "total": paired,
            "accuracy": actual[direction] / paired if paired else None,
        }
        for direction in ("POSITIVE", "NEGATIVE", "NEUTRAL")
    }
    relative = [
        row for row in outcomes
        if row.get("relativeDirectionVerdict") in {"CORRECT", "WRONG"}
    ]
    return {
        "units": "target-observations",
        "scoredTargets": paired,
        "modelDirectionMatches": correct,
        "modelDirectionalAccuracy": correct / paired if paired else None,
        "directionalPredictions": len(directional),
        "directionalMatches": sum(row["status"] == "CORRECT" for row in directional),
        "directionalAccuracy": (
            sum(row["status"] == "CORRECT" for row in directional) / len(directional)
            if directional else None
        ),
        "actualDirectionDistribution": dict(actual),
        "predictedDirectionDistribution": dict(predicted),
        "constantDirectionControls": baselines,
        "beatsEveryConstantControl": (
            correct > max(actual.values()) if paired else None
        ),
        "relativeScoredTargets": len(relative),
        "relativeMatches": sum(
            row["relativeDirectionVerdict"] == "CORRECT" for row in relative
        ),
        "relativeAccuracy": (
            sum(row["relativeDirectionVerdict"] == "CORRECT" for row in relative)
            / len(relative) if relative else None
        ),
        "attributionEstablished": False,
        "limitation": (
            "Controls compare price directions on the SAME evaluated assets; "
            "they do not prove causal attribution or a tradable signal. "
            "Neutral is based on the legacy +/-0.1% absolute-price rule."
        ),
    }


def holdout_selection(journal: Path, source_db: Path, count: int) -> tuple[list[dict], dict]:
    if not 1 <= count <= 1000:
        raise ValueError("Holdout count must be 1..1000")
    if journal.is_symlink() or source_db.is_symlink():
        raise ValueError("Symlink inputs are not allowed")
    if not journal.is_file() or not source_db.is_file():
        raise ValueError("Both original training journal and news archive must exist")
    with sqlite3.connect(journal.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.execute("PRAGMA query_only=ON")
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
    holdout, reserved = load_articles(
        source_db, 1000, cutoff, partition="holdout", **kwargs,
    )
    last_training = timestamp(bounds[0])
    selected = [
        a for a in holdout if a["articleId"] not in training_ids
        and publication_time(a) > last_training
    ][:count]
    info = {
        "trainingJournal": str(journal),
        "holdoutSourceDb": str(source_db),
        "trainingArticleCount": bounds[1],
        "lastTrainingPublishedAt": last_training.isoformat(),
        "holdoutReserved": reserved,
        "holdoutEligible": len(holdout),
        "holdoutSelected": len(selected),
        "sources": dict(sorted(Counter(row["source"] for row in selected).items())),
        "earliestRelease": publication_time(selected[0]).isoformat() if selected else None,
        "latestRelease": publication_time(selected[-1]).isoformat() if selected else None,
        "trainingOverlap": False,
        "pointInTimeCertified": False,
        "learnedRulesApplied": 0,
        "learningDisabled": True,
        "assessmentMode": "FROZEN_NO_FEEDBACK_NOT_TRAINED_MODEL_A_B",
    }
    return selected, {**settings, "holdoutInfo": info}


def evaluate_holdout(
    *, journal: Path, source_db: Path, output: Path, model: object,
    client: httpx.Client, count: int = 100,
) -> dict:
    if output.exists() or output.is_symlink():
        raise ValueError("Refusing to overwrite a holdout result")
    selected, settings = holdout_selection(journal, source_db, count)
    last_training = timestamp(settings["holdoutInfo"]["lastTrainingPublishedAt"])
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
    comparison = price_direction_comparison(outcomes)
    result = {
        "schemaVersion": "frozen_chronological_holdout_v2",
        "cohort": settings["holdoutInfo"],
        "priceDirectionControls": comparison,
        "trainingJournal": str(journal), "holdoutSourceDb": str(source_db),
        "pointInTimeCertified": False, "independentSourceVerified": False,
        "learningDisabled": True, "feedbackRulesApplied": 0,
        "lastTrainingPublishedAt": last_training.isoformat(),
        "holdoutSelected": len(selected),
        "holdoutCompleted": sum(x["status"] == "COMPLETE" for x in outcomes),
        "holdoutWithScoredTargets": sum(any(
            row.get("status") in {"CORRECT", "WRONG"}
            for row in x["evaluation"]["impacts"]
        ) for x in outcomes),
        "holdoutWithoutTargets": sum(not x["prediction"]["impacts"] for x in outcomes),
        "holdoutModelErrors": sum(x["status"] == "SKIPPED_INVALID_MODEL" for x in outcomes),
        "targetsPriceScored": len(judged),
        "directionalPriceScored": len(directional),
        "directionalPriceMatches": sum(
            x["status"] == "CORRECT" for x in directional
        ),
        "causalAttributionEstablished": False,
        "warning": ("Retrospective content is not PIT-certified. Model prompting "
                    "uses no learned feedback. This evaluates a frozen baseline, "
                    "not an independently verified gain from training. "
                    "The train/holdout source material may be retrospectively enriched."),
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
    parser.add_argument("--output", type=Path)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--preview", action="store_true",
                        help="Inspect reserved holdout counts without Gemma or price requests")
    args = parser.parse_args()
    if args.preview:
        _, settings = holdout_selection(args.journal, args.source_db, args.count)
        print(json.dumps(settings["holdoutInfo"], ensure_ascii=False))
        return
    if args.output is None:
        parser.error("--output is required unless --preview is provided")
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
