"""Offline, CPU-only NEWS importance candidate research on protected splits.

No inference from stock-price movement, no generated human labels, no TEST
gold access in training/evaluation, no automatic production model promotion.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import unicodedata
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.news_importance_evaluation import _metrics
from tris_market_event_intelligence.news_importance_review import (
    _cohort,
    _instant,
    _read,
    score_review,
)
from tris_market_event_intelligence.story_index import WORD

ALPHAS = (0.25, 0.5, 1.0, 2.0, 4.0)
CLASSES = ("IMPORTANT", "UNIMPORTANT")
MIN_TRAIN_PER_CLASS = 3
MIN_VALIDATION_PER_CLASS = 2
TABLE = """CREATE TABLE IF NOT EXISTS news_research_experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_sha256 TEXT NOT NULL UNIQUE,
    cohort_sha256 TEXT NOT NULL,
    plan_sha256 TEXT NOT NULL,
    review_sha256 TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    alpha REAL NOT NULL,
    outcome TEXT NOT NULL,
    train_count INTEGER NOT NULL,
    validation_count INTEGER NOT NULL,
    candidate_log_loss REAL,
    baseline_log_loss REAL,
    finished_at TEXT NOT NULL
)"""


def _tokens(article: dict[str, object]) -> list[str]:
    title = article["title"]
    summary = article.get("summary") or ""
    return [
        token for token in WORD.findall(
            unicodedata.normalize("NFKC", title + " " + summary).casefold()
        ) if len(token) >= 2
    ][:500]


def _validate_inputs(
    cohort_path: Path, plan_path: Path, review_path: Path,
) -> tuple[list[tuple[list[str], str]], list[tuple[list[str], str]], dict[str, str]]:
    cohort_raw, cohort = _cohort(cohort_path)
    score_review(cohort_path, review_path)
    review_raw = _read(review_path)
    reviews = json.loads(review_raw)
    plan_raw = _read(plan_path)
    plan = json.loads(plan_raw)
    cohort_sha = hashlib.sha256(cohort_raw).hexdigest()
    if not isinstance(plan, dict) or (
        plan.get("schemaVersion") != "news_holdout_plan_v1"
        or plan.get("cohortSha256") != cohort_sha
        or plan.get("status") != "PLANNED_NOT_AUDITED"
        or plan.get("stockPricesUsed") is not False
        or plan.get("trainingDatasetEligible") is not False
        or plan.get("modelTrained") is not False
        or plan.get("humanReviewTimeGateApplied") is not True
        or plan.get("humanReviewSha256") != hashlib.sha256(review_raw).hexdigest()
        or plan.get("humanGoldValuesExposed") is not False
    ):
        raise ValueError("Human-review time-gated news holdout required")
    articles = {entry["articleId"]: entry for entry in cohort["items"]}
    labels = {entry["articleId"]: entry for entry in reviews["judgments"]}
    if not isinstance(plan.get("items"), list) or len(plan["items"]) != len(articles):
        raise ValueError("Incomplete research partition list")
    observed: set[int] = set()
    counts: Counter[str] = Counter()
    train, validation = [], []
    train_end = _instant(plan["trainEnd"])
    for item in plan["items"]:
        if not isinstance(item, dict):
            raise TypeError("Malformed partition row")
        identifier = item.get("articleId")
        if type(identifier) is not int or identifier not in articles or identifier in observed:
            raise ValueError("Missing or repeated original article ID")
        observed.add(identifier)
        article = articles[identifier]
        if item.get("textSha256") != article["textSha256"] or (
            _instant(item["firstObservedAt"]) != _instant(article["firstObservedAt"])
        ):
            raise ValueError("Research partition article provenance mismatch")
        partition = item.get("split")
        if partition not in {"TRAIN", "VALIDATION", "TEST", "EMBARGO", "QUARANTINED"}:
            raise ValueError("Unexpected research partition")
        counts[partition] += 1
        human = labels[identifier]
        if partition == "TRAIN":
            status = item.get("trainGoldAvailability")
            if status == "LABEL_AVAILABLE_BY_TRAIN_END_NOT_AUDITED":
                if human["importance"] not in CLASSES or (
                    _instant(human["reviewedAt"]) > train_end
                ):
                    raise ValueError("Training label unavailable at cutoff")
                train.append((_tokens(article), human["importance"]))
            elif status not in {
                "UNREVIEWED", "HUMAN_UNCERTAIN", "REVIEWED_AFTER_TRAIN_END",
            }:
                raise ValueError("Unrecognized training gold availability")
        elif item.get("trainGoldAvailability") != "NOT_TRAIN_LABEL":
            raise ValueError("Protected research partition has ambiguous gold access")
        if partition == "VALIDATION" and human["importance"] in CLASSES:
            validation.append((_tokens(article), human["importance"]))
        # TEST is deliberately never read for labels or prediction metrics.
    if dict(counts) != plan.get("counts") or observed != set(articles):
        raise ValueError("Research plan article counts do not match cohort")
    hashes = {
        "cohortSha256": cohort_sha,
        "planSha256": hashlib.sha256(plan_raw).hexdigest(),
        "reviewSha256": hashlib.sha256(review_raw).hexdigest(),
    }
    return train, validation, hashes


def _train_naive_bayes(
    train: list[tuple[list[str], str]], alpha: float,
):
    counts = {label: Counter() for label in CLASSES}
    documents = Counter()
    for words, label in train:
        counts[label].update(words)
        documents[label] += 1
    vocabulary = set(counts[CLASSES[0]]) | set(counts[CLASSES[1]])
    size = max(1, len(vocabulary))
    totals = {label: sum(counts[label].values()) + alpha * size for label in CLASSES}

    def predict(words: list[str]) -> dict[str, float]:
        frequencies = Counter(word for word in words if word in vocabulary)
        logits = {
            label: math.log((documents[label] + alpha) / (len(train) + alpha * 2))
            + sum(
                amount * math.log((counts[label][word] + alpha) / totals[label])
                for word, amount in frequencies.items()
            )
            for label in CLASSES
        }
        maximum = max(logits.values())
        exp = {label: math.exp(logits[label] - maximum) for label in CLASSES}
        norm = sum(exp.values())
        return {
            "IMPORTANT": exp["IMPORTANT"] / norm,
            "UNIMPORTANT": exp["UNIMPORTANT"] / norm,
            "UNCERTAIN": 0.0,
        }

    return predict


def _journal(path: Path) -> sqlite3.Connection:
    if path.is_symlink() or path.is_dir() or (
        not path.exists() and not path.parent.is_dir()
    ):
        raise ValueError("Safe research-only SQLite path in existing directory required")
    uri = path.resolve().as_uri() + "?mode=rwc"
    db = sqlite3.connect(uri, uri=True, timeout=10, isolation_level=None)
    db.execute("PRAGMA busy_timeout=10000")
    db.execute(TABLE)
    return db


def research_news_once(
    *, cohort_path: Path, plan_path: Path, review_path: Path,
    journal_path: Path,
) -> dict[str, object]:
    """Run at most one new alpha experiment; never train on inferred labels.

    Repeated calls exhaust the finite alpha grid; no fabricated improvement
    and no hot loop when the archive has no new or independently labeled data.
    """
    train, validation, hashes = _validate_inputs(cohort_path, plan_path, review_path)
    amounts = {
        "train": Counter(label for _, label in train),
        "validation": Counter(label for _, label in validation),
    }
    eligible = all(
        amounts[group][label] >= count
        for group, count in (("train", MIN_TRAIN_PER_CLASS),
                             ("validation", MIN_VALIDATION_PER_CLASS))
        for label in CLASSES
    )
    with closing(_journal(journal_path)) as db:
        db.execute("BEGIN IMMEDIATE")
        try:
            if not eligible:
                db.execute("COMMIT")
                return {
                    "schemaVersion": "news_research_iteration_v1",
                    "status": "BLOCKED_INSUFFICIENT_HUMAN_GOLD",
                    "trainingArticles": len(train),
                    "validationArticles": len(validation),
                    "minimumTrainPerClass": MIN_TRAIN_PER_CLASS,
                    "minimumValidationPerClass": MIN_VALIDATION_PER_CLASS,
                    "modelTrained": False,
                    "stockPricesUsed": False,
                    "productionDbModified": False,
                }
            candidate = None
            experiment_sha = None
            for alpha in ALPHAS:
                fingerprint = json.dumps({
                    **hashes, "algorithm": "multinomial_naive_bayes_v1",
                    "alpha": alpha,
                }, sort_keys=True)
                sha = hashlib.sha256(fingerprint.encode()).hexdigest()
                exists = db.execute(
                    "SELECT 1 FROM news_research_experiments WHERE experiment_sha256=?",
                    (sha,),
                ).fetchone()
                if not exists:
                    candidate, experiment_sha = alpha, sha
                    break
            if candidate is None:
                db.execute("COMMIT")
                return {
                    "schemaVersion": "news_research_iteration_v1",
                    "status": "EXPERIMENT_GRID_EXHAUSTED",
                    "trainingArticles": len(train),
                    "validationArticles": len(validation),
                    "modelTrained": False,
                    "newResearchExperiment": False,
                    "stockPricesUsed": False,
                    "productionDbModified": False,
                }
            model = _train_naive_bayes(train, candidate)
            scored = [(label, model(words)) for words, label in validation]
            baseline = [(label, {
                "IMPORTANT": 0.0, "UNIMPORTANT": 1.0, "UNCERTAIN": 0.0,
            }) for _, label in validation]
            model_metric = _metrics(scored)["meanLogLoss"]
            baseline_metric = _metrics(baseline)["meanLogLoss"]
            outcome = "DESCRIPTIVE_RESEARCH_ONLY"
            db.execute(
                "INSERT INTO news_research_experiments VALUES "
                "(NULL,?,?,?,?,?,?,?,?,?,?)",
                (experiment_sha, hashes["cohortSha256"], hashes["planSha256"],
                 hashes["reviewSha256"], "multinomial_naive_bayes_v1", candidate,
                 outcome, len(train), len(validation), model_metric,
                 baseline_metric, datetime.now(UTC).isoformat()),
            )
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
    return {
        "schemaVersion": "news_research_iteration_v1",
        "status": outcome,
        "experimentSha256": experiment_sha,
        "algorithm": "multinomial_naive_bayes_v1",
        "smoothingAlpha": candidate,
        "trainingArticles": len(train),
        "validationArticles": len(validation),
        "candidateValidationLogLoss": model_metric,
        "alwaysUnimportantValidationLogLoss": baseline_metric,
        "newResearchExperiment": True,
        "modelTrainedInMemory": True,
        "modelPersistedOrPromoted": False,
        "humanGoldIndependentlyAudited": False,
        "holdoutNeverExposedExternallyVerified": False,
        "modelImprovementValidated": False,
        "testPartitionEvaluated": False,
        "stockPricesUsed": False,
        "productionDbModified": False,
    }


def audit_research_journal(journal_path: Path) -> dict[str, object]:
    """Inspect only research metadata; cannot expose original news text."""
    with closing(_journal(journal_path)) as db:
        records = db.execute(
            "SELECT experiment_sha256,alpha,outcome,train_count,validation_count,"
            "candidate_log_loss,baseline_log_loss FROM news_research_experiments "
            "ORDER BY id"
        ).fetchall()
    return {
        "schemaVersion": "news_research_journal_audit_v1",
        "iterations": [{
            "experimentSha256": row[0], "alpha": row[1], "outcome": row[2],
            "trainingArticles": row[3], "validationArticles": row[4],
            "candidateValidationLogLoss": row[5],
            "baselineValidationLogLoss": row[6],
        } for row in records],
        "totalIterations": len(records),
        "modelImprovementValidated": False,
        "testGoldEvaluated": False,
        "stockPricesUsed": False,
    }
