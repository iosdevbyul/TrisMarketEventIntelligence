"""Conservative source-to-actor boundary checks for agency event hypotheses.

These checks target documented category mistakes. Passing them does not
establish economic causality, validity, or point-in-time source availability.
"""
from __future__ import annotations

import re

_NO_TARGET = frozenset({"none", "null", "n/a", "not applicable", "no target"})
_FAA_COST = re.compile(
    r"\b(?:compliance|inspection|maintenance|operational|grounding)\s+"
    r"(?:costs?|expenses?|burden|restrictions?)\s+"
    r"(?:on|for|borne by|incurred by)\s+boeing\b",
    re.IGNORECASE,
)
_FAA_COST_REVERSE = re.compile(
    r"\bboeing\b.{0,70}?\b(?:must|incurs?|pays?|bears?|absorbs?)\b"
    r".{0,50}?\b(?:airline|inspection|compliance|grounding)\b",
    re.IGNORECASE,
)
_FAA_OPERATOR = re.compile(
    r"\b(?:airlines?|operators?|airplanes?\s+operated\s+by)\b",
    re.IGNORECASE,
)
_COMMERCIAL_ASSURANCE = re.compile(
    r"\b(?:guarantee[ds]?|will\s+(?:increase|generate|earn)|"
    r"proven\s+(?:revenue|profit)|certain\s+(?:sales|return))\b",
    re.IGNORECASE,
)


def target_is_abstention_sentinel(target: str) -> bool:
    """An explicit null-target is not a supported corporate hypothesis."""
    return target.strip().casefold() in _NO_TARGET


def inspect_evidence_boundary(proposal: dict, case: dict) -> dict:
    """Bound effects to primary-source actors and prevent certainty laundering."""
    reasons: list[str] = []
    warnings: list[str] = []
    facts = {row["factId"]: row["statement"] for row in case["facts"]}
    cited = " ".join(facts[fid] for fid in proposal["sourceFactIds"] if fid in facts)
    explanation = proposal["mechanismJustification"]
    combined = " ".join([
        explanation,
        proposal["conditionalEffect"],
        proposal["evidenceBoundary"],
    ])
    if (
        case["category"] == "AVIATION_SAFETY_REGULATION"
        and proposal["target"] == "Boeing"
        and _FAA_OPERATOR.search(cited)
        and (
            _FAA_COST.search(explanation) or _FAA_COST_REVERSE.search(explanation)
        )
    ):
        # FAA grounded airplanes operated by airlines. This source alone
        # does not establish the economic incidence of airlines' costs on
        # the manufacturer, even when a downstream impact is plausible.
        reasons.append("UNSUPPORTED_OPERATOR_TO_MANUFACTURER_COST_ATTRIBUTION")
    if _COMMERCIAL_ASSURANCE.search(combined):
        reasons.append("UNDOCUMENTED_COMMERCIAL_OUTCOME_ASSERTED")
    if (
        case["category"] == "PUBLIC_HEALTH_REGULATION"
        and proposal["transmission"] == "REGULATION"
        and "authorization" in cited.casefold()
    ):
        warnings.append("AUTHORIZATION_IS_NOT_EVIDENCE_OF_REALIZED_SALES")
    return {
        "blockers": reasons,
        "warnings": warnings,
        "documentedActorAndEconomicCostIncidenceVerified": False,
        "semanticAccuracyVerified": False,
        "humanReviewRequired": True,
    }


def describe_abstention_format(case: dict, proposals: list[dict]) -> dict:
    """Keep model abstention vs output format separate; never inflate success."""
    negative = case["category"] == "SCIENTIFIC_MISSION_NEGATIVE_CONTROL"
    has_sentinel = any(
        target_is_abstention_sentinel(row.get("target", ""))
        for row in proposals
    )
    return {
        "negativeControl": negative,
        "modelAbstainedWithEmptyHypotheses": negative and not proposals,
        "modelExpressedNoTargetViaSentinel": negative and has_sentinel,
        "abstentionOutputFormatViolation": negative and has_sentinel,
        "negativeControlPass": negative and not proposals,
        "semanticAccuracyVerified": False,
    }
