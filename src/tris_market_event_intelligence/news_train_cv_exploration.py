"""TRAIN-only, bounded repeatable NEWS importance feature exploration.

Historical news can support multiple *different* hypotheses without new
articles. The reserved TEST set and VALIDATION labels are NEVER scored here;
training-only stratified folds evaluate each candidate. Repeated folds are
not new independent evidence or proof of accuracy improvement.
"""
from __future__ import annotations

import hashlib
import json
import math
from contextlib import closing
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

from tris_market_event_intelligence.news_importance_evaluation import _metrics
from tris_market_event_intelligence.news_naive_bayes_model import NewsImportanceModel
from tris_market_event_intelligence.news_research_worker import (
    CLASSES,
    MIN_TRAIN_PER_CLASS,
    _journal,
    _validate_inputs,
)

ALPHAS = (0.25, 1.0, 4.0)
FEATURES = ("UNIGRAM", "UNIGRAM_BIGRAM")
TOKEN_LIMITS = (32, 128, 500)
MAX_FOLDS = 3
EXPLORATION_VERSION = "news_train_cv_exploration_v1"
ALGORITHM = "train_only_multinomial_nb_cv_v1"
TABLE = """CREATE TABLE IF NOT EXISTS news_train_cv_experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_sha256 TEXT NOT NULL UNIQUE,
    cohort_sha256 TEXT NOT NULL,
    plan_sha256 TEXT NOT NULL,
    review_sha256 TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    feature_mode TEXT NOT NULL,
    token_limit INTEGER NOT NULL,
    alpha REAL NOT NULL,
    fold_count INTEGER NOT NULL,
    training_count INTEGER NOT NULL,
    cv_mean_log_loss REAL NOT NULL,
    finished_at TEXT NOT NULL
)"""


def _features(words: list[str], *, mode: str, token_limit: int) -> list[str]:
    if mode not in FEATURES or token_limit not in TOKEN_LIMITS:
        raise ValueError("Unrecognized bounded NEWS feature configuration")
    # Token lengths capped to support the existing model state constraints;
    # all transformations are deterministic and depend on NEWS text only.
    tokens = [word for word in words[:token_limit] if 2 <= len(word) <= 64]
    if mode == "UNIGRAM":
        return tokens
    return tokens + [
        f"{a}~{b}" for a, b in pairwise(tokens)
    ]


def _folds(
    train: list[tuple[list[str], str]],
) -> list[tuple[list[int], list[int]]]:
    by_class: dict[str, list[int]] = {label: [] for label in CLASSES}
    for index, (_, label) in enumerate(train):
        if label not in CLASSES:
            raise ValueError("TRAIN-only CV requires genuine binary importance labels")
        by_class[label].append(index)
    if any(len(by_class[label]) < MIN_TRAIN_PER_CLASS for label in CLASSES):
        return []
    fold_count = min(MAX_FOLDS, *(len(by_class[label]) for label in CLASSES))
    partitions: list[list[int]] = [[] for _ in range(fold_count)]
    for label in CLASSES:
        for position, identifier in enumerate(by_class[label]):
            partitions[position % fold_count].append(identifier)
    total = set(range(len(train)))
    return [
        (sorted(total.difference(fold)), sorted(fold))
        for fold in partitions
    ]


def _score(
    train: list[tuple[list[str], str]], *, mode: str,
    token_limit: int, alpha: float,
) -> tuple[float, int]:
    partitions = _folds(train)
    if not partitions:
        raise ValueError("Insufficient human-labeled TRAIN rows for stratified CV")
    weighted_log_loss = 0.0
    evaluated = 0
    for training_indices, held_indices in partitions:
        fold_train = [
            (_features(train[index][0], mode=mode, token_limit=token_limit),
             train[index][1])
            for index in training_indices
        ]
        model = NewsImportanceModel.from_training(fold_train, alpha)
        predictions = [
            (train[index][1], model.predict(
                _features(train[index][0], mode=mode, token_limit=token_limit)
            ))
            for index in held_indices
        ]
        loss = _metrics(predictions)["meanLogLoss"]
        if not math.isfinite(loss):
            raise ValueError("Nonfinite TRAIN-only CV log loss")
        weighted_log_loss += loss * len(held_indices)
        evaluated += len(held_indices)
    return weighted_log_loss / evaluated, len(partitions)


def explore_news_train_once(
    *, cohort_path: Path, plan_path: Path, review_path: Path,
    journal_path: Path,
) -> dict[str, object]:
    """Run at most one previously untried experiment on TRAIN folds only.

    Existing alpha-only research journal and model artifacts are untouched.
    This independent exploratory table must never be treated as promotion
    evidence. VALIDATION/TEST outcomes are not scored or selected upon.
    """
    train, _validation, hashes = _validate_inputs(
        cohort_path, plan_path, review_path,
    )
    if not _folds(train):
        return {
            "schemaVersion": EXPLORATION_VERSION,
            "status": "BLOCKED_INSUFFICIENT_TRAIN_HUMAN_GOLD",
            "trainingArticles": len(train),
            "modelTrained": False,
            "validationGoldScored": False,
            "testGoldScored": False,
            "productionModelChanged": False,
            "stockPricesUsed": False,
        }
    with closing(_journal(journal_path)) as db:
        db.execute(TABLE)
        db.execute("BEGIN IMMEDIATE")
        try:
            choice = None
            experiment_sha = None
            for mode in FEATURES:
                for limit in TOKEN_LIMITS:
                    for alpha in ALPHAS:
                        fingerprint = json.dumps({
                            **hashes, "algorithm": ALGORITHM,
                            "featureMode": mode, "tokenLimit": limit,
                            "alpha": alpha,
                        }, sort_keys=True)
                        sha = hashlib.sha256(fingerprint.encode()).hexdigest()
                        if db.execute(
                            "SELECT 1 FROM news_train_cv_experiments "
                            "WHERE experiment_sha256=?", (sha,),
                        ).fetchone() is None:
                            choice = (mode, limit, alpha)
                            experiment_sha = sha
                            break
                    if choice is not None:
                        break
                if choice is not None:
                    break
            if choice is None:
                db.execute("COMMIT")
                return {
                    "schemaVersion": EXPLORATION_VERSION,
                    "status": "TRAIN_CV_SEARCH_SPACE_EXHAUSTED",
                    "trainingArticles": len(train),
                    "newResearchExperiment": False,
                    "modelTrained": False,
                    "validationGoldScored": False,
                    "testGoldScored": False,
                    "productionModelChanged": False,
                    "stockPricesUsed": False,
                }
            mode, limit, alpha = choice
            cv_loss, folds = _score(train, mode=mode, token_limit=limit, alpha=alpha)
            db.execute(
                "INSERT INTO news_train_cv_experiments "
                "(experiment_sha256,cohort_sha256,plan_sha256,review_sha256,"
                "algorithm,feature_mode,token_limit,alpha,fold_count,"
                "training_count,cv_mean_log_loss,finished_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (experiment_sha, hashes["cohortSha256"], hashes["planSha256"],
                 hashes["reviewSha256"], ALGORITHM, mode, limit, alpha,
                 folds, len(train), cv_loss, datetime.now(UTC).isoformat()),
            )
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
    return {
        "schemaVersion": EXPLORATION_VERSION,
        "status": "TRAIN_ONLY_CV_EXPLORATORY",
        "experimentSha256": experiment_sha,
        "algorithm": ALGORITHM,
        "featureMode": mode,
        "tokenLimit": limit,
        "smoothingAlpha": alpha,
        "trainingArticles": len(train),
        "trainOnlyFoldCount": folds,
        "trainOnlyCrossValidationLogLoss": cv_loss,
        "newResearchExperiment": True,
        "validationGoldScored": False,
        "testGoldScored": False,
        "humanGoldIndependentlyAudited": False,
        "modelImprovementValidated": False,
        "modelPersistedOrPromoted": False,
        "productionModelChanged": False,
        "stockPricesUsed": False,
    }


def audit_train_exploration(journal_path: Path) -> dict[str, object]:
    with closing(_journal(journal_path)) as db:
        db.execute(TABLE)
        records = db.execute(
            "SELECT experiment_sha256,feature_mode,token_limit,alpha,"
            "fold_count,training_count,cv_mean_log_loss "
            "FROM news_train_cv_experiments ORDER BY id"
        ).fetchall()
    return {
        "schemaVersion": "news_train_cv_exploration_audit_v1",
        "status": "RESEARCH_ONLY_NOT_INDEPENDENT_VALIDATION",
        "totalExperiments": len(records),
        "items": [{
            "experimentSha256": row[0], "featureMode": row[1],
            "tokenLimit": row[2], "smoothingAlpha": row[3],
            "foldCount": row[4], "trainingArticles": row[5],
            "trainOnlyCrossValidationLogLoss": row[6],
        } for row in records],
        "trainingOnlyMetrics": True,
        "validationGoldScored": False,
        "testGoldScored": False,
        "modelImprovementValidated": False,
        "productionModelChanged": False,
        "stockPricesUsed": False,
    }
