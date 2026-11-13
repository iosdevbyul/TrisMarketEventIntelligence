"""Conservative, source-linked triage, never a fact-certification engine."""

from dataclasses import asdict, dataclass

HIGH_REVIEW_CATEGORIES = frozenset({
    "MONETARY_POLICY", "ECONOMIC_INDICATOR", "REGULATION",
    "GEOPOLITICAL", "SUPPLY_CHAIN",
})


@dataclass(frozen=True)
class ClaimAssessment:
    claim_id: int
    verification_level: str
    latest_review_status: str | None
    review_revision: int
    review_priority: str
    priority_reason: str
    fact_confirmed: bool
    analysis_suitability: str


def assess_claim(
    *, claim_id: int, classification: str, status: str,
    claim_kind: str, polarity: str, temporal_context: str,
    review_status: str | None, revision: int,
) -> dict[str, object]:
    """Review-priority heuristics are not economic significance or event truth."""
    if review_status in {"RETRACTED", "DISPUTED"}:
        level, priority, reason, suitability = (
            review_status, "REVIEW_FIRST", "correction_or_dispute",
            "EXCLUDE_PENDING_REVIEW",
        )
    elif review_status == "CONFIRMED":
        level, priority, reason, suitability = (
            "HUMAN_REVIEWED_PRIMARY_DOCUMENT", "NORMAL",
            "primary_document_review_recorded", "HUMAN_REVIEWED_RESEARCH",
        )
    elif review_status == "CORROBORATED":
        level, priority, reason, suitability = (
            "HUMAN_REVIEWED_CORROBORATION", "REVIEW_FIRST",
            "corroboration_is_not_confirmation", "CONDITIONAL_RESEARCH",
        )
    else:
        level = (
            "RULE_SUPPORTED_NOT_INDEPENDENTLY_VERIFIED"
            if status == "VERIFIED" else "NOT_INDEPENDENTLY_VERIFIED"
        )
        suitability = "CONDITIONAL_RESEARCH"
        if claim_kind != "REPORTED" or polarity != "AFFIRMED":
            priority, reason = "DEFER", "non_reported_or_non_affirmative"
            suitability = "EXCLUDE_PENDING_REVIEW"
        elif temporal_context == "PAST":
            priority, reason = "DEFER", "historical_reference"
            suitability = "EXCLUDE_PENDING_REVIEW"
        elif classification in HIGH_REVIEW_CATEGORIES:
            priority, reason = "REVIEW_FIRST", "category_review_heuristic"
        else:
            priority, reason = "NORMAL", "standard_review_queue"
    if review_status == "UNDER_REVIEW":
        level = "HUMAN_REVIEW_IN_PROGRESS"
        if priority == "DEFER":
            priority, reason = "NORMAL", "human_review_in_progress"
    return asdict(ClaimAssessment(
        claim_id, level, review_status, revision, priority, reason,
        False, suitability,
    ))
