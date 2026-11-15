"""Fail-closed, TRAIN-only candidate selection lock for NEWS research.

This is NOT a model artifact, model deployment approval, or an independent
evaluation. Once created, the manifest pins the single selected hypothesis
and source hashes; the existing TEST usage ledger remains the only cooperative
mechanism for one final TEST reservation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from contextlib import closing
from pathlib import Path

from tris_market_event_intelligence.news_candidate_artifact import _canonical
from tris_market_event_intelligence.news_importance_review import _read
from tris_market_event_intelligence.news_research_policy import (
    MAX_HYPOTHESES,
    TABLE,
    _catalog,
)
from tris_market_event_intelligence.news_research_worker import _journal, _validate_inputs
from tris_market_event_intelligence.news_test_usage_ledger import audit_news_test_usage

SCHEMA = "news_train_candidate_lock_v1"
MAX_MANIFEST_BYTES = 16384


def _read_locked(path: Path) -> dict[str, object]:
    raw = _read(path)
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError("Candidate lock exceeds size budget")
    document = json.loads(raw)
    if not isinstance(document, dict) or set(document) != {
        "payload", "payloadSha256",
    }:
        raise ValueError("Invalid candidate lock wrapper")
    payload = document["payload"]
    if not isinstance(payload, dict) or set(payload) != {
        "schemaVersion", "experimentSha256", "cohortSha256",
        "planSha256", "reviewSha256", "algorithm", "featureMode",
        "tokenCap", "alpha", "trainCvLogLoss", "catalogCount",
        "selectionRule", "validationGoldScored", "testGoldScored",
        "independentImprovementVerified", "promotionAllowed",
        "stockPricesUsed",
    } or payload["schemaVersion"] != SCHEMA or any(
        payload[key] is not False for key in (
            "validationGoldScored", "testGoldScored",
            "independentImprovementVerified", "promotionAllowed",
            "stockPricesUsed",
        )
    ):
        raise ValueError("Unsafe candidate lock schema")
    if hashlib.sha256(_canonical(payload)).hexdigest() != document["payloadSha256"]:
        raise ValueError("Candidate lock digest mismatch")
    for key in ("experimentSha256", "cohortSha256", "planSha256", "reviewSha256"):
        sha = payload[key]
        if not isinstance(sha, str) or len(sha) != 64 or any(
            char not in "0123456789abcdef" for char in sha
        ):
            raise ValueError("Invalid candidate lock fingerprint")
    if payload["algorithm"] != "train_only_feature_hypothesis_cv_v1" or (
        payload["featureMode"] not in ("WORD_PRESENCE", "CHAR_TRIGRAM_PRESENCE")
    ) or payload["selectionRule"] != "minimum_train_cv_log_loss_then_sha" or (
        type(payload["catalogCount"]) is not int
        or payload["catalogCount"] != MAX_HYPOTHESES
        or type(payload["tokenCap"]) is not int
        or type(payload["alpha"]) not in (int, float)
        or type(payload["trainCvLogLoss"]) not in (int, float)
        or not math.isfinite(payload["trainCvLogLoss"])
        or payload["trainCvLogLoss"] < 0
    ):
        raise ValueError("Invalid bounded candidate selection metadata")
    return document


def select_train_candidate(*, cohort_path: Path, plan_path: Path,
                           review_path: Path, journal_path: Path) -> dict[str, object]:
    """Verify full, untampered TRAIN-only catalog before locking one hypothesis."""
    _train, _validation, hashes = _validate_inputs(
        cohort_path, plan_path, review_path,
    )
    expected = {
        sha: (mode, cap, alpha)
        for sha, mode, cap, alpha in _catalog(hashes)
    }
    with closing(_journal(journal_path)) as db:
        db.execute(TABLE)
        rows = db.execute(
            "SELECT experiment_sha256,feature_mode,token_cap,alpha,"
            "train_cv_log_loss FROM news_research_policy_experiments "
            "WHERE cohort_sha256=? AND plan_sha256=? AND review_sha256=?",
            (hashes["cohortSha256"], hashes["planSha256"], hashes["reviewSha256"]),
        ).fetchall()
    if len(rows) != MAX_HYPOTHESES:
        raise ValueError("All 12 pinned TRAIN-only hypotheses must complete first")
    seen = set()
    for sha, mode, cap, alpha, loss in rows:
        if sha in seen or sha not in expected or (mode, cap, alpha) != expected[sha]:
            raise ValueError("Tampered or duplicate TRAIN-only hypothesis journal")
        if type(loss) not in (int, float) or not math.isfinite(loss) or loss < 0:
            raise ValueError("Invalid recorded TRAIN CV score")
        seen.add(sha)
    winner = min(rows, key=lambda row: (row[4], row[0]))
    sha, mode, cap, alpha, loss = winner
    payload = {
        "schemaVersion": SCHEMA,
        **hashes,
        "experimentSha256": sha,
        "algorithm": "train_only_feature_hypothesis_cv_v1",
        "featureMode": mode, "tokenCap": cap, "alpha": alpha,
        "trainCvLogLoss": loss, "catalogCount": MAX_HYPOTHESES,
        "selectionRule": "minimum_train_cv_log_loss_then_sha",
        "validationGoldScored": False, "testGoldScored": False,
        "independentImprovementVerified": False,
        "promotionAllowed": False, "stockPricesUsed": False,
    }
    return {"payload": payload, "payloadSha256": hashlib.sha256(
        _canonical(payload)
    ).hexdigest()}


def create_candidate_lock(*, output_path: Path, cohort_path: Path,
                          plan_path: Path, review_path: Path,
                          journal_path: Path) -> dict[str, object]:
    """Create private immutable-by-convention manifest, never overwrite."""
    if output_path.is_symlink() or output_path.exists() or not output_path.parent.is_dir():
        raise ValueError("New ordinary candidate lock in existing directory required")
    wrapper = select_train_candidate(
        cohort_path=cohort_path, plan_path=plan_path,
        review_path=review_path, journal_path=journal_path,
    )
    blob = _canonical(wrapper) + b"\n"
    if len(blob) > MAX_MANIFEST_BYTES:
        raise ValueError("Candidate lock exceeds size budget")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(output_path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(blob)
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    return {
        "schemaVersion": SCHEMA, "status": "TRAIN_CANDIDATE_LOCKED_NOT_VALIDATED",
        "experimentSha256": wrapper["payload"]["experimentSha256"],
        "manifestSha256": hashlib.sha256(blob).hexdigest(),
        "validationGoldScored": False, "testGoldScored": False,
        "promotionAllowed": False, "stockPricesUsed": False,
    }


def audit_candidate_lock(*, lock_path: Path, usage_ledger_path: Path | None = None,
                         cohort_path: Path | None = None,
                         plan_path: Path | None = None,
                         review_path: Path | None = None,
                         journal_path: Path | None = None) -> dict[str, object]:
    """Audit lock, optional current source and existing final TEST reservations."""
    wrapper = _read_locked(lock_path)
    payload = wrapper["payload"]
    provided = (cohort_path, plan_path, review_path, journal_path)
    if any(p is not None for p in provided):
        if not all(p is not None for p in provided):
            raise ValueError("All pinned research input paths required for replay audit")
        current = select_train_candidate(
            cohort_path=cohort_path, plan_path=plan_path,
            review_path=review_path, journal_path=journal_path,
        )
        if current != wrapper:
            raise ValueError("Candidate lock differs from pinned TRAIN journal")
    reservations = []
    if usage_ledger_path is not None:
        ledger = audit_news_test_usage(usage_ledger_path)
        reservations = [
            item for item in ledger["reservations"]
            if item["cohortSha256"] == payload["cohortSha256"]
            and item["purpose"] == "TEST_FINAL"
        ]
    return {
        "schemaVersion": "news_train_candidate_lock_audit_v1",
        "status": "RESEARCH_ONLY_INDEPENDENT_EVALUATION_PENDING",
        "experimentSha256": payload["experimentSha256"],
        "pinnedInputsReplayed": all(p is not None for p in provided),
        "finalTestReservationsInCooperativeLedger": len(reservations),
        "finalTestAccessOutsideLedgerVerified": False,
        "goldIndependentlyAudited": False,
        "independentImprovementVerified": False,
        "promotionAllowed": False, "stockPricesUsed": False,
    }
