"""Finite NEWS-only hypothesis research policy, independent of validation gold.

A bounded, immutable hypothesis catalog diversifies feature representations.
Identical journal entries are never retrained; correlated TRAIN-fold scores
are exploratory evidence only, never an independent promotion gate.
"""
from __future__ import annotations

import hashlib
import json
import math
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from tris_market_event_intelligence.news_importance_evaluation import _metrics
from tris_market_event_intelligence.news_naive_bayes_model import NewsImportanceModel
from tris_market_event_intelligence.news_research_worker import _journal, _validate_inputs
from tris_market_event_intelligence.news_train_cv_exploration import _folds

VERSION = "news_research_policy_v1"
ALGORITHM = "train_only_feature_hypothesis_cv_v1"
MODES = ("WORD_PRESENCE", "CHAR_TRIGRAM_PRESENCE")
CAPS = (32, 128)
ALPHAS = (0.25, 1.0, 4.0)
MAX_HYPOTHESES = len(MODES) * len(CAPS) * len(ALPHAS)
TABLE = """CREATE TABLE IF NOT EXISTS news_research_policy_experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_sha256 TEXT NOT NULL UNIQUE,
    cohort_sha256 TEXT NOT NULL,
    plan_sha256 TEXT NOT NULL,
    review_sha256 TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    feature_mode TEXT NOT NULL,
    token_cap INTEGER NOT NULL,
    alpha REAL NOT NULL,
    folds INTEGER NOT NULL,
    train_count INTEGER NOT NULL,
    train_cv_log_loss REAL NOT NULL,
    completed_at TEXT NOT NULL
)"""


def _representation(words: list[str], mode: str, cap: int) -> list[str]:
    if mode not in MODES or cap not in CAPS:
        raise ValueError("Unrecognized bounded NEWS hypothesis")
    selected = [w for w in words[:cap] if 2 <= len(w) <= 64]
    if mode == "WORD_PRESENCE":
        return list(dict.fromkeys(selected))
    grams: dict[str, None] = {}
    for word in selected:
        for i in range(max(1, min(len(word) - 2, 8))):
            grams["#" + word[i:i + 3]] = None
    return list(grams)


def _cv_score(train: list[tuple[list[str], str]], mode: str,
              cap: int, alpha: float) -> tuple[float, int]:
    folds = _folds(train)
    if not folds:
        raise ValueError("Insufficient independent TRAIN human labels")
    total, evaluated = 0.0, 0
    for fit_ids, score_ids in folds:
        fitted = [
            (_representation(train[i][0], mode, cap), train[i][1])
            for i in fit_ids
        ]
        model = NewsImportanceModel.from_training(fitted, alpha)
        predictions = [
            (train[i][1], model.predict(_representation(train[i][0], mode, cap)))
            for i in score_ids
        ]
        loss = _metrics(predictions)["meanLogLoss"]
        if not math.isfinite(loss):
            raise ValueError("Nonfinite TRAIN-only evaluation")
        total += loss * len(score_ids)
        evaluated += len(score_ids)
    return total / evaluated, len(folds)


def _catalog(hashes: dict[str, str]):
    for mode in MODES:
        for cap in CAPS:
            for alpha in ALPHAS:
                config = {
                    **hashes, "algorithm": ALGORITHM,
                    "featureMode": mode, "tokenCap": cap, "alpha": alpha,
                }
                fingerprint = hashlib.sha256(
                    json.dumps(config, sort_keys=True).encode("utf-8")
                ).hexdigest()
                yield fingerprint, mode, cap, alpha


def run_policy_once(*, cohort_path: Path, plan_path: Path,
                    review_path: Path, journal_path: Path) -> dict[str, object]:
    train, _unused_validation, hashes = _validate_inputs(
        cohort_path, plan_path, review_path,
    )
    safe = {
        "schemaVersion": VERSION,
        "validationGoldScored": False,
        "testGoldScored": False,
        "independentImprovementVerified": False,
        "promotionAllowed": False,
        "productionModelChanged": False,
        "stockPricesUsed": False,
    }
    if not _folds(train):
        return {**safe, "status": "BLOCKED_INSUFFICIENT_TRAIN_HUMAN_GOLD",
                "newResearchExperiment": False}
    with closing(_journal(journal_path)) as db:
        db.execute(TABLE)
        db.execute("BEGIN IMMEDIATE")
        try:
            # For this pinned source, never exceed the catalog size even
            # if a process is restarted while the same DB remains in use.
            completed = {
                row[0] for row in db.execute(
                    "SELECT experiment_sha256 FROM news_research_policy_experiments"
                )
            }
            candidate = next(
                ((sha, mode, cap, alpha) for sha, mode, cap, alpha
                 in _catalog(hashes) if sha not in completed), None,
            )
            if candidate is None:
                db.execute("COMMIT")
                return {**safe, "status": "POLICY_HYPOTHESES_EXHAUSTED",
                        "newResearchExperiment": False,
                        "candidateSelectionValidated": False}
            sha, mode, cap, alpha = candidate
            score, fold_count = _cv_score(train, mode, cap, alpha)
            db.execute(
                "INSERT INTO news_research_policy_experiments "
                "(experiment_sha256,cohort_sha256,plan_sha256,review_sha256,"
                "algorithm,feature_mode,token_cap,alpha,folds,train_count,"
                "train_cv_log_loss,completed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (sha, hashes["cohortSha256"], hashes["planSha256"],
                 hashes["reviewSha256"], ALGORITHM, mode, cap, alpha,
                 fold_count, len(train), score, datetime.now(UTC).isoformat()),
            )
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
    return {
        **safe, "status": "TRAIN_ONLY_HYPOTHESIS_RESEARCH",
        "experimentSha256": sha, "featureMode": mode, "tokenCap": cap,
        "smoothingAlpha": alpha, "trainFoldCount": fold_count,
        "trainFoldLogLoss": score, "newResearchExperiment": True,
        "candidateSelectionValidated": False,
    }


def audit_policy(journal_path: Path) -> dict[str, object]:
    with closing(_journal(journal_path)) as db:
        db.execute(TABLE)
        rows = db.execute(
            "SELECT experiment_sha256,feature_mode,token_cap,alpha,"
            "train_cv_log_loss FROM news_research_policy_experiments "
            "ORDER BY id"
        ).fetchall()
    return {
        "schemaVersion": "news_research_policy_audit_v1",
        "status": "TRAIN_ONLY_RESEARCH_NOT_INDEPENDENT_VALIDATION",
        "totalExperiments": len(rows),
        "catalogSizePerPinnedInput": MAX_HYPOTHESES,
        "researchOnlyRanking": [
            {"experimentSha256": sha, "featureMode": mode,
             "tokenCap": cap, "alpha": alpha, "trainFoldLogLoss": loss}
            for sha, mode, cap, alpha, loss in
            sorted(rows, key=lambda row: (row[4], row[0]))
        ],
        "validationGoldScored": False, "testGoldScored": False,
        "independentImprovementVerified": False,
        "promotionAllowed": False, "stockPricesUsed": False,
    }
