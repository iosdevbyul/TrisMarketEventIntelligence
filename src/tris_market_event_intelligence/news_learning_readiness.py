"""Read-only readiness gate for privately human-reviewed NEWS research.

Summarizes source integrity, TRAIN label sufficiency and cross-split lexical
review candidates. No independent truth certification or model promotion.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from tris_market_event_intelligence.news_near_duplicate_audit import (
    audit_news_near_duplicates,
)
from tris_market_event_intelligence.news_research_worker import (
    CLASSES,
    MIN_TRAIN_PER_CLASS,
    _validate_inputs,
)
from tris_market_event_intelligence.news_train_candidate_lock import (
    audit_candidate_lock,
)

SCHEMA = "news_learning_readiness_v1"


def inspect_news_learning_readiness(
    *, cohort_path: Path, plan_path: Path, review_path: Path,
    lock_path: Path | None = None, journal_path: Path | None = None,
    usage_ledger_path: Path | None = None,
) -> dict[str, object]:
    """Inspect pinned inputs and possible leakage, without evaluating gold.

    Does not score VALIDATION or TEST; human review validation is a schema
    and provenance check, not independent verification of label accuracy.
    """
    if (lock_path is None) != (journal_path is None):
        raise ValueError("Candidate lock and journal must be provided together")
    if usage_ledger_path is not None and lock_path is None:
        raise ValueError("TEST ledger audit requires a pinned candidate lock")
    train, _protected_validation, hashes = _validate_inputs(
        cohort_path, plan_path, review_path,
    )
    duplicates = audit_news_near_duplicates(
        cohort_path=cohort_path, holdout_path=plan_path,
    )
    counts = Counter(label for _, label in train)
    enough = all(counts[label] >= MIN_TRAIN_PER_CLASS for label in CLASSES)
    candidates = duplicates["crossPartitionCandidates"]
    lock = None
    if lock_path is not None:
        lock = audit_candidate_lock(
            lock_path=lock_path, usage_ledger_path=usage_ledger_path,
            cohort_path=cohort_path, plan_path=plan_path,
            review_path=review_path, journal_path=journal_path,
        )
    issues = []
    if not enough:
        issues.append("INSUFFICIENT_TRAIN_HUMAN_GOLD")
    if candidates:
        issues.append("CROSS_SPLIT_LEXICAL_REVIEW_REQUIRED")
    if lock is not None and lock["finalTestReservationsInCooperativeLedger"]:
        issues.append("FINAL_TEST_ALREADY_RESERVED")
    if not issues:
        status = "PRELIMINARY_RESEARCH_READY_NOT_INDEPENDENTLY_AUDITED"
    else:
        status = "BLOCKED_REVIEW_OR_RESEARCH_PREREQUISITES"
    return {
        "schemaVersion": SCHEMA,
        "status": status,
        **hashes,
        "trainingArticlesWithTimeGatedHumanLabels": len(train),
        "trainingLabelCounts": {label: counts[label] for label in CLASSES},
        "minimumTrainLabelsPerClass": MIN_TRAIN_PER_CLASS,
        "trainFoldResearchFeasible": enough,
        "crossSplitLexicalReviewCandidates": candidates,
        "lexicalAuditStatus": duplicates["status"],
        "lexicalAuditThreshold": duplicates["threshold"],
        "lockedCandidateVerifiedAgainstJournal": (
            lock["pinnedInputsReplayed"] if lock is not None else False
        ),
        "finalTestReservationsInCooperativeLedger": (
            lock["finalTestReservationsInCooperativeLedger"]
            if lock is not None else None
        ),
        "blockingFindings": issues,
        "noLexicalCandidatesProvesNoLeakage": False,
        "semanticTranslationAndCrossPageLeakageCleared": False,
        "humanGoldIndependentlyAdjudicated": False,
        "validationGoldScored": False,
        "testGoldScored": False,
        "independentImprovementVerified": False,
        "promotionAllowed": False,
        "productionModelChanged": False,
        "stockPricesUsed": False,
    }
