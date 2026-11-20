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

from tris_market_event_intelligence.distributed_holdout_selection import (
    spread_date_selection,
)
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
from tris_market_event_intelligence.market_direction_calibration import (
    calibrate_direction,
    fit_price_calibrator,
)
from tris_market_event_intelligence.market_event_candidate_policy import (
    filter_market_event_candidates,
    research_candidate,
)
from tris_market_event_intelligence.market_prior_baseline import (
    annotate_price_study,
    forecast_targets,
)
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
    paired_baseline = [
        row for row in outcomes if row.get("baselinePriceMatch") is not None
    ]
    paired_combo = [
        row for row in outcomes if row.get("combinedPriceMatch") is not None
    ]
    paired_gated = [
        row for row in outcomes if row.get("gatedPriceMatch") is not None
    ]
    gated_reasons = Counter(
        (row.get("gatedNewsAdjustment") or {}).get("reasonCode", "UNKNOWN")
        for row in outcomes if row.get("gatedNewsAdjustment")
    )
    paired_learned = [
        row for row in outcomes if row.get("frozenCalibrationMatch") is not None
    ]
    by_week = Counter(
        str(case.get("releaseAt", ""))[:7] for case in cases
        if case.get("releaseAt")
    )
    return {
        "units": "target-observations",
        "monthDistribution": dict(sorted(by_week.items())),
        "multipleMonthsObserved": len(by_week) >= 2,
        "priorBaseline": {
            "compared": len(paired_baseline),
            "matches": sum(x["baselinePriceMatch"] for x in paired_baseline),
            "accuracy": (sum(x["baselinePriceMatch"] for x in paired_baseline)
                         / len(paired_baseline) if paired_baseline else None),
        },
        "trainedCalibration": {
            "compared": len(paired_learned),
            "matches": sum(x["frozenCalibrationMatch"] for x in paired_learned),
            "accuracy": (sum(x["frozenCalibrationMatch"] for x in paired_learned)
                         / len(paired_learned) if paired_learned else None),
            "rawModelMatchesOnSameTargets": sum(
                x.get("status") == "CORRECT" for x in paired_learned
            ),
            "baselineMatchesOnSameTargets": sum(
                x.get("baselinePriceMatch") is True for x in paired_learned
            ),
        },
        "evidenceGatedNews": {
            "compared": len(paired_gated),
            "matches": sum(row["gatedPriceMatch"] for row in paired_gated),
            "accuracy": (sum(row["gatedPriceMatch"] for row in paired_gated)
                         / len(paired_gated) if paired_gated else None),
            "newsChangedForecastCount": sum(
                row["gatedNewsAdjustment"].get("newsAdjusted") is True
                for row in paired_gated
            ),
            "reasonCounts": dict(sorted(gated_reasons.items())),
            "baselineMatchesOnSameTargets": sum(
                row["baselinePriceMatch"] is True for row in paired_gated
            ),
        },
        "baselinePlusNews": {
            "compared": len(paired_combo),
            "matches": sum(x["combinedPriceMatch"] for x in paired_combo),
            "accuracy": (sum(x["combinedPriceMatch"] for x in paired_combo)
                         / len(paired_combo) if paired_combo else None),
        },
        "pairedModelAgainstBaseline": {
            "compared": len(paired_baseline),
            "modelMatches": sum(x.get("status") == "CORRECT"
                                for x in paired_baseline),
            "baselineMatches": sum(x["baselinePriceMatch"]
                                   for x in paired_baseline),
            "modelWins": sum(x.get("status") == "CORRECT"
                             and not x["baselinePriceMatch"] for x in paired_baseline),
            "baselineWins": sum(x.get("status") == "WRONG"
                                and x["baselinePriceMatch"] for x in paired_baseline),
        },
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


def holdout_selection(
    journal: Path, source_db: Path, count: int, *,
    selection_policy: str = "chronological",
    exclude_report: Path | list[Path] | None = None,
    inventory_as_of: str | None = None,
) -> tuple[list[dict], dict]:
    if not 1 <= count <= 1000:
        raise ValueError("Holdout count must be 1..1000")
    if selection_policy not in {"chronological", "spread-dates",
                              "event-spread-dates"}:
        raise ValueError("Unknown holdout selection policy")
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
    if inventory_as_of is not None:
        revised = timestamp(inventory_as_of)
        if revised < cutoff:
            raise ValueError("INVENTORY_CUTOFF_CANNOT_PRECEDE_TRAINING")
        from datetime import UTC, datetime

        if revised > datetime.now(UTC):
            raise ValueError("FUTURE_INVENTORY_CUTOFF_NOT_ALLOWED")
        cutoff = revised
    cache = settings.get("sourceContentCache")
    kwargs = {"content_cache": Path(cache)} if cache else {}
    holdout, reserved = load_articles(
        source_db, 1000, cutoff,
        partition="all" if selection_policy in {
            "spread-dates", "event-spread-dates"
        } else "holdout",
        **kwargs,
    )
    excluded_previous: set[int] = set()
    previous_reports = (
        [exclude_report] if isinstance(exclude_report, Path)
        else list(exclude_report or [])
    )
    for prior_path in previous_reports:
        if prior_path.is_symlink() or not prior_path.is_file():
            raise ValueError("Missing or unsafe previous Holdout report")
        previous = json.loads(prior_path.read_text(encoding="utf-8"))
        if (previous.get("holdoutSourceDb") != str(source_db.resolve())
                or not str(previous.get("schemaVersion", "")).startswith(
                    "frozen_chronological_holdout_"
                )):
            raise ValueError("Previous Holdout provenance mismatch")
        excluded_previous.update(
            int(item["articleId"]) for item in previous.get("articles", [])
        )
    training_fingerprints = {
        (row.get("textSha256") or row.get("sourceUrl"))
        for row in holdout if row["articleId"] in training_ids
    }
    training_fingerprints.discard(None)
    last_training = timestamp(bounds[0])
    eligible = [
        a for a in holdout
        if a["articleId"] not in training_ids
        and a["articleId"] not in excluded_previous
        and publication_time(a) > last_training
        and (a.get("textSha256") or a.get("sourceUrl")) not in training_fingerprints
    ]
    candidate_audit = None
    if selection_policy == "event-spread-dates":
        eligible, candidate_audit = filter_market_event_candidates(eligible)
    distinct_info = None
    if selection_policy in {"spread-dates", "event-spread-dates"}:
        selected, distinct_info = spread_date_selection(eligible, count)
        if candidate_audit is not None:
            for item in distinct_info["items"]:
                article = next(
                    row for row in selected if row["articleId"] == item["articleId"]
                )
                item["marketEventEligibility"] = research_candidate(article)
    else:
        selected = eligible[:count]
    info = {
        "trainingJournal": str(journal),
        "holdoutSourceDb": str(source_db),
        "inventoryAsOf": cutoff.isoformat(),
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
        "selectionPolicy": selection_policy,
        "candidateScope": (
            "UNTRAINED_POST_TRAINING_ARCHIVE_NOT_ORIGINAL_HOLDOUT"
            if selection_policy in {"spread-dates", "event-spread-dates"}
            else "STRICT_TRAILING_HOLDOUT"
        ),
        "previouslyEvaluatedArticleIdsExcluded": len(excluded_previous),
        "dateSpread": distinct_info,
        "economicEventScreen": candidate_audit,
    }
    return selected, {**settings, "holdoutInfo": info}


def evaluate_holdout(
    *, journal: Path, source_db: Path, output: Path, model: object,
    client: httpx.Client, count: int = 100,
    selection_policy: str = "chronological",
    exclude_report: Path | list[Path] | None = None,
    selection_manifest: Path | None = None,
    inventory_as_of: str | None = None,
) -> dict:
    if output.exists() or output.is_symlink():
        raise ValueError("Refusing to overwrite a holdout result")
    selected, settings = holdout_selection(
        journal, source_db, count, selection_policy=selection_policy,
        exclude_report=exclude_report,
        inventory_as_of=inventory_as_of,
    )
    if selection_policy in {"spread-dates", "event-spread-dates"} and len(selected) != count:
        available = settings["holdoutInfo"]["dateSpread"]["candidateDistinctDates"]
        raise ValueError(
            f"INSUFFICIENT_DISTINCT_RELEASE_DATES: requested {count}, "
            f"eligible {available}; collect more qualifying historical news"
        )
    if selection_manifest is not None:
        if selection_manifest.is_symlink() or not selection_manifest.is_file():
            raise ValueError("Missing or unsafe frozen article manifest")
        saved = json.loads(selection_manifest.read_text(encoding="utf-8"))
        current = settings["holdoutInfo"]
        if (saved.get("inventoryAsOf") != current["inventoryAsOf"]
                or saved.get("holdoutSourceDb") != current["holdoutSourceDb"]
                or saved.get("selectionPolicy") != selection_policy
                or saved.get("dateSpread") != current["dateSpread"]
                or saved.get("economicEventScreen") != current["economicEventScreen"]
                or saved.get("previouslyEvaluatedArticleIdsExcluded")
                != current["previouslyEvaluatedArticleIdsExcluded"]):
            raise ValueError("FROZEN_ARTICLE_MANIFEST_MISMATCH")
    last_training = timestamp(settings["holdoutInfo"]["lastTrainingPublishedAt"])
    # Fit once on sealed training journal only. No holdout labels enter fitting.
    fitted = (
        fit_price_calibrator(journal, publication_time(selected[0]))
        if selected else {"status": "NO_HOLDOUT", "ruleCount": 0, "rules": {}}
    )
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
            # Commit every pre-event price baseline and the news-adjusted
            # direction before fetching any of the post-release closes.
            precommitted = forecast_targets(
                prediction, client, observed,
                market_context=context["marketContext"],
            )
            prior_calibration = [
                calibrate_direction(fitted, str(row.get("priceProxy", "")),
                                    row["direction"])
                for row in prediction["impacts"]
            ]
            price = annotate_price_study(
                evaluate(prediction, observed, horizon, settings["interval"], client),
                precommitted,
            )
            for assessed, calibrated in zip(
                price["impacts"], prior_calibration, strict=True,
            ):
                assessed["frozenCalibration"] = calibrated
                assessed["frozenCalibrationMatch"] = (
                    calibrated["direction"] == assessed.get("actualDirection")
                    if calibrated["status"] == "AVAILABLE"
                    and assessed.get("status") in {"CORRECT", "WRONG"} else None
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
        "schemaVersion": "frozen_chronological_holdout_v4",
        "cohort": settings["holdoutInfo"],
        "eventCandidateSelectionUsedOutcomes": False,
        "priceDirectionControls": comparison,
        "trainedPriceCalibration": fitted,
        "marketRegimeDiversityVerified": False,
        "moreThanOneCalendarMonthObserved": comparison["multipleMonthsObserved"],
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
        "pastOnlyBaselineEnabled": True,
        "warning": ("Retrospective content is not PIT-certified. Model prompting "
                    "uses no learned feedback. This evaluates a frozen baseline, "
                    "not an independently verified gain from training. "
                    "The train/holdout source material may be retrospectively enriched. "
                    "Calibration is historical price coincidence, not model fine tuning. "
                    "Distinct release dates do not establish independent market regimes. "
                    "Evidence gating is a conservative safety policy, not proof of alpha."),
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
    parser.add_argument(
        "--selection-policy",
        choices=["chronological", "spread-dates", "event-spread-dates"],
        default="chronological",
        help="spread-dates requires distinct original release dates",
    )
    parser.add_argument(
        "--exclude-report", type=Path, action="append",
        help="Avoid reusing article IDs already evaluated in a previous Holdout JSON",
    )
    parser.add_argument(
        "--inventory-as-of",
        help="ISO-8601 cutoff for newly captured sources; never changes training",
    )
    parser.add_argument("--selection-manifest", type=Path,
                        help="Refuse to run if candidate list differs from prior preview")
    parser.add_argument("--manifest", type=Path,
                        help="Write immutable selected-article list during preview")
    parser.add_argument("--preview", action="store_true",
                        help="Inspect reserved holdout counts without Gemma or price requests")
    args = parser.parse_args()
    if args.preview:
        if args.selection_manifest is not None:
            parser.error("--selection-manifest is for the actual evaluation run")
        _, settings = holdout_selection(
            args.journal, args.source_db, args.count,
            selection_policy=args.selection_policy,
            exclude_report=args.exclude_report,
            inventory_as_of=args.inventory_as_of,
        )
        report = settings["holdoutInfo"]
        if args.manifest is not None:
            if args.manifest.exists() or args.manifest.is_symlink():
                parser.error("Refusing to overwrite article selection manifest")
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            with args.manifest.open("x", encoding="utf-8") as output:
                json.dump(report, output, ensure_ascii=False, indent=2)
        print(json.dumps(report, ensure_ascii=False))
        return
    if args.output is None:
        parser.error("--output is required unless --preview is provided")
    if args.manifest is not None:
        parser.error("--manifest is supported only with --preview")
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
            selection_policy=args.selection_policy,
            exclude_report=args.exclude_report,
            selection_manifest=args.selection_manifest,
            inventory_as_of=args.inventory_as_of,
        ), ensure_ascii=False))


if __name__ == "__main__":
    main()
