"""Private, fail-closed comparison of news-only research model candidates.

This produces a human review packet, NEVER a production deployment decision.
Repeated VALIDATION tuning, unverified human gold and reserved TEST are not
independent evidence of model generalization or safe promotion.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from tris_market_event_intelligence.news_candidate_artifact import (
    compare_candidate_replay,
    read_candidate_artifact,
)
from tris_market_event_intelligence.news_importance_evaluation import _metrics
from tris_market_event_intelligence.news_importance_review import _read
from tris_market_event_intelligence.news_research_worker import _validate_inputs

MAX_CANDIDATES = 16
MIN_REVIEWED_VALIDATION = 100
MIN_EACH_CLASS = 20


def compare_news_candidates(
    *, candidate_paths: list[Path], cohort_path: Path, plan_path: Path,
    review_path: Path, journal_path: Path,
) -> dict[str, object]:
    """Rank replayed candidates by descriptive validation log loss only.

    No independent human evaluation or holdout access is performed here.
    Even a winning model is not eligible for automatic promotion.
    """
    if not isinstance(candidate_paths, list) or not (
        2 <= len(candidate_paths) <= MAX_CANDIDATES
    ) or len({str(path.resolve()) for path in candidate_paths}) != len(candidate_paths):
        raise ValueError("Provide 2..16 distinct private model candidates")
    train, validation, hashes = _validate_inputs(
        cohort_path, plan_path, review_path,
    )
    if not train or not validation:
        raise ValueError("News train and validation must be nonempty")
    original_news = _read(cohort_path)
    original_plan = _read(plan_path)
    original_review = _read(review_path)
    if any(
        hashlib.sha256(raw).hexdigest() != hashes[key]
        for raw, key in (
            (original_news, "cohortSha256"),
            (original_plan, "planSha256"),
            (original_review, "reviewSha256"),
        )
    ):
        raise ValueError("Research source hashes differ from validated inputs")
    counts = Counter(label for _, label in validation)
    baseline_probs = {
        "IMPORTANT": 0.0, "UNIMPORTANT": 1.0, "UNCERTAIN": 0.0,
    }
    baseline = _metrics([
        (label, baseline_probs) for _, label in validation
    ])
    summary: list[dict[str, object]] = []
    seen_experiments: set[str] = set()
    for path in candidate_paths:
        replay = compare_candidate_replay(
            candidate_path=path, cohort_path=cohort_path,
            plan_path=plan_path, review_path=review_path,
            journal_path=journal_path,
        )
        model, payload = read_candidate_artifact(path)
        sha = payload["experimentSha256"]
        if sha in seen_experiments:
            raise ValueError("Duplicate news research experiment in comparison")
        seen_experiments.add(sha)
        if any(payload[key] != hashes[key] for key in hashes):
            raise ValueError("News model research provenance mismatch")
        if replay["status"] != "REPRODUCIBLE_RESEARCH_ONLY":
            raise ValueError("Only fully replayed research candidates permitted")
        measurements = _metrics([
            (label, model.predict(words)) for words, label in validation
        ])
        if not math.isclose(
            measurements["meanLogLoss"], payload["validationLogLoss"],
            rel_tol=1e-12, abs_tol=1e-12,
        ):
            raise ValueError("Model's recorded validation result changed")
        summary.append({
            "experimentSha256": sha,
            "artifactSha256": hashlib.sha256(_read(path)).hexdigest(),
            "smoothingAlpha": model.alpha,
            "validationArticles": measurements["n"],
            "validationLogLoss": measurements["meanLogLoss"],
            "validationMacroF1": measurements["macroF1"],
            "validationBrier": measurements["multiclassBrier"],
            "importantPrecision": measurements["importantPrecision"],
            "importantRecall": measurements["importantRecall"],
            "replayVerified": True,
            "independentGoldVerified": False,
        })
    summary.sort(key=lambda row: (
        row["validationLogLoss"], row["experimentSha256"],
    ))
    screened = bool(
        len(validation) >= MIN_REVIEWED_VALIDATION
        and all(counts[label] >= MIN_EACH_CLASS for label in (
            "IMPORTANT", "UNIMPORTANT",
        ))
    )
    best = summary[0]
    beats_baseline = bool(
        best["validationLogLoss"] < baseline["meanLogLoss"]
    )
    return {
        "schemaVersion": "news_candidate_comparison_v1",
        "status": (
            "DESCRIPTIVE_SCREENING_ONLY" if screened
            else "INSUFFICIENT_INDEPENDENT_NEWS_GOLD"
        ),
        **hashes,
        "candidateCount": len(summary),
        "trainingArticles": len(train),
        "validationArticles": len(validation),
        "validationClassCounts": dict(sorted(counts.items())),
        "minimumValidationArticlesForScreening": MIN_REVIEWED_VALIDATION,
        "minimumValidationPerClass": MIN_EACH_CLASS,
        "alwaysUnimportantBaseline": {
            "logLoss": baseline["meanLogLoss"],
            "macroF1": baseline["macroF1"],
            "multiclassBrier": baseline["multiclassBrier"],
        },
        "rankedResearchCandidates": summary,
        "descriptivelyBestExperimentSha256": best["experimentSha256"],
        "bestCandidateBeatsSimpleBaselineOnValidation": beats_baseline,
        "rankingIsIndependentPerformanceProof": False,
        "goldIndependentlyAdjudicated": False,
        "nearDuplicateLeakageIndependentlyCleared": False,
        "holdoutNeverUsedForTuningVerified": False,
        "protectedTestAccessed": False,
        "humanApprovalRecorded": False,
        "promotionAllowed": False,
        "rollbackNeeded": False,
        "productionModelChanged": False,
        "modelImprovementValidated": False,
        "stockPricesUsed": False,
    }


def write_comparison_report(path: Path, result: dict[str, object]) -> None:
    """Write one private report, never altering an existing artifact."""
    from tris_market_event_intelligence.news_candidate_artifact import _canonical
    import os

    if path.is_symlink() or path.exists() or not path.parent.is_dir():
        raise ValueError("Private report needs a new file in an existing directory")
    blob = _canonical(result) + b"\n"
    if len(blob) > 1_000_000:
        raise ValueError("Research report exceeds maximum size")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(blob)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
