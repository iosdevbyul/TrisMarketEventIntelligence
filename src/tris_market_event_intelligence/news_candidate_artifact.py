"""Export and reload private, provenance-pinned news classifier research artifacts.

Only a previously journaled research candidate may be exported. Its recorded
VALIDATION loss is recomputed before any create-only artifact is written.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from contextlib import closing
from pathlib import Path

from tris_market_event_intelligence.news_importance_evaluation import _metrics
from tris_market_event_intelligence.news_importance_review import _read
from tris_market_event_intelligence.news_naive_bayes_model import NewsImportanceModel
from tris_market_event_intelligence.news_research_worker import (
    _journal,
    _tokens,
    _validate_inputs,
)

SCHEMA = "news_research_candidate_artifact_v1"
MAX_ARTIFACT_BYTES = 8_000_000


def _canonical(data: object) -> bytes:
    return json.dumps(
        data, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def build_candidate_artifact(
    *, cohort_path: Path, plan_path: Path, review_path: Path,
    journal_path: Path, experiment_sha256: str,
) -> dict[str, object]:
    """Rebuild only a journaled candidate, refusing any metric discrepancy."""
    if not isinstance(experiment_sha256, str) or len(experiment_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in experiment_sha256
    ):
        raise ValueError("Exact research experiment SHA required")
    train, validation, hashes = _validate_inputs(
        cohort_path, plan_path, review_path,
    )
    with closing(_journal(journal_path)) as db:
        record = db.execute(
            "SELECT cohort_sha256,plan_sha256,review_sha256,algorithm,alpha,"
            "train_count,validation_count,candidate_log_loss,baseline_log_loss "
            "FROM news_research_experiments WHERE experiment_sha256=?",
            (experiment_sha256,),
        ).fetchone()
    if record is None or tuple(hashes.values()) != tuple(record[:3]) or (
        record[3] != "multinomial_naive_bayes_v1"
        or len(train) != record[5] or len(validation) != record[6]
        or not all(type(x) in (int, float) and math.isfinite(x) for x in record[7:9])
    ):
        raise ValueError("Completed research record and immutable inputs do not match")
    alpha = record[4]
    # Existing research journal IDs were generated from json.dumps(sort_keys=True)
    # using default spacing, not our artifact serialization.
    existing_fingerprint = json.dumps({
        **hashes, "algorithm": "multinomial_naive_bayes_v1",
        "alpha": alpha,
    }, sort_keys=True)
    if hashlib.sha256(existing_fingerprint.encode()).hexdigest() != experiment_sha256:
        raise ValueError("Research experiment fingerprint mismatch")
    model = NewsImportanceModel.from_training(train, alpha)
    metrics = _metrics([(label, model.predict(words)) for words, label in validation])
    baseline = _metrics([(label, {
        "IMPORTANT": 0.0, "UNIMPORTANT": 1.0, "UNCERTAIN": 0.0,
    }) for _, label in validation])
    if not (
        math.isclose(metrics["meanLogLoss"], record[7], rel_tol=1e-12, abs_tol=1e-12)
        and math.isclose(
            baseline["meanLogLoss"], record[8], rel_tol=1e-12, abs_tol=1e-12
        )
    ):
        raise ValueError("Recreated model differs from recorded validation metrics")
    payload = {
        "schemaVersion": SCHEMA,
        "experimentSha256": experiment_sha256,
        "algorithm": "multinomial_naive_bayes_v1",
        **hashes,
        "modelState": model.to_state(),
        "trainingArticles": len(train),
        "validationArticles": len(validation),
        "validationLogLoss": metrics["meanLogLoss"],
        "productionEligible": False,
        "humanGoldIndependentlyVerified": False,
        "protectedTestEvaluated": False,
        "newsOnly": True,
        "stockPricesUsed": False,
    }
    blob = _canonical(payload)
    if len(blob) > MAX_ARTIFACT_BYTES:
        raise ValueError("Candidate exceeds private artifact size limit")
    return {"payload": payload, "payloadSha256": hashlib.sha256(blob).hexdigest()}


def write_candidate_artifact(path: Path, artifact: dict[str, object]) -> None:
    """Exclusive create; never overwrite an existing model, even on a retry."""
    if path.is_symlink() or path.exists() or not path.parent.is_dir():
        raise ValueError("Private model artifact must be a new ordinary file")
    encoded = _canonical(artifact) + b"\n"
    if len(encoded) > MAX_ARTIFACT_BYTES:
        raise ValueError("Model artifact exceeds size limit")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def read_candidate_artifact(path: Path) -> tuple[NewsImportanceModel, dict[str, object]]:
    """Load untrusted JSON, validate all fields and the canonical payload SHA."""
    raw = _read(path)
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise ValueError("Candidate artifact exceeds size limit")
    wrapper = json.loads(raw)
    if not isinstance(wrapper, dict) or set(wrapper) != {
        "payload", "payloadSha256",
    }:
        raise ValueError("Invalid model artifact wrapper")
    payload = wrapper["payload"]
    if not isinstance(payload, dict) or set(payload) != {
        "schemaVersion", "experimentSha256", "algorithm",
        "cohortSha256", "planSha256", "reviewSha256", "modelState",
        "trainingArticles", "validationArticles", "validationLogLoss",
        "productionEligible", "humanGoldIndependentlyVerified",
        "protectedTestEvaluated", "newsOnly", "stockPricesUsed",
    } or payload["schemaVersion"] != SCHEMA or (
        payload["algorithm"] != "multinomial_naive_bayes_v1"
        or payload["productionEligible"] is not False
        or payload["newsOnly"] is not True
        or payload["stockPricesUsed"] is not False
        or payload["protectedTestEvaluated"] is not False
        or payload["humanGoldIndependentlyVerified"] is not False
    ):
        raise ValueError("Untrusted model candidate schema or promotion flags")
    if any(not isinstance(payload[key], str) or len(payload[key]) != 64 or any(
        char not in "0123456789abcdef" for char in payload[key]
    ) for key in (
        "experimentSha256", "cohortSha256", "planSha256", "reviewSha256"
    )):
        raise ValueError("Invalid research provenance SHA")
    if any(type(payload[key]) is not int or not 1 <= payload[key] <= 5000
           for key in ("trainingArticles", "validationArticles")):
        raise ValueError("Invalid training/validation sizes")
    loss = payload["validationLogLoss"]
    if type(loss) not in (float, int) or not math.isfinite(loss) or loss < 0:
        raise ValueError("Invalid validation loss")
    if hashlib.sha256(_canonical(payload)).hexdigest() != wrapper["payloadSha256"]:
        raise ValueError("Research model artifact content SHA mismatch")
    model = NewsImportanceModel.from_state(payload["modelState"])
    if sum(model.documents.values()) != payload["trainingArticles"]:
        raise ValueError("Stored document totals do not match research metadata")
    return model, payload


def compare_candidate_replay(
    *, candidate_path: Path, cohort_path: Path, plan_path: Path,
    review_path: Path, journal_path: Path,
) -> dict[str, object]:
    """Verify deterministic reloaded predictions against recomputed training."""
    restored, payload = read_candidate_artifact(candidate_path)
    recreated = build_candidate_artifact(
        cohort_path=cohort_path, plan_path=plan_path,
        review_path=review_path, journal_path=journal_path,
        experiment_sha256=payload["experimentSha256"],
    )
    if recreated["payload"] != payload:
        raise ValueError("Persisted model weights differ from recreated training model")
    train, validation, hashes = _validate_inputs(cohort_path, plan_path, review_path)
    if any(payload[key] != hashes[key] for key in hashes):
        raise ValueError("Artifact research provenance mismatch")
    fresh = NewsImportanceModel.from_training(train, restored.alpha)
    for words, _ in validation:
        for label in ("IMPORTANT", "UNIMPORTANT", "UNCERTAIN"):
            if restored.predict(words)[label] != fresh.predict(words)[label]:
                raise ValueError("Reloaded model prediction is not reproducible")
    return {
        "schemaVersion": "news_candidate_replay_v1",
        "status": "REPRODUCIBLE_RESEARCH_ONLY",
        "experimentSha256": payload["experimentSha256"],
        "validationPredictionsCompared": len(validation),
        "modelQualityImprovementValidated": False,
        "productionEligible": False,
        "testGoldAccessed": False,
        "stockPricesUsed": False,
    }
