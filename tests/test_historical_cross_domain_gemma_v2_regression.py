"""Observed Gemma4 v2 FAA/FDA/NASA scenario regression cases."""
from __future__ import annotations

import copy
import json

from test_historical_cross_domain_official_study import FakeProvider, cohort

from tris_market_event_intelligence import historical_cross_domain_official_study_cli as cross
from tris_market_event_intelligence.historical_cross_domain_semantic_boundary import (
    inspect_evidence_boundary,
)
from tris_market_event_intelligence.historical_official_conditional_scenarios import (
    review_scenario,
)


def _fda_proposal(target: str) -> dict:
    return {
        "target": target,
        "sourceFactIds": ["fda-authorization"],
        "transmission": "REGULATION",
        "direction": "POSITIVE",
        "mechanismJustification": (
            f"IF the updated authorization leads to increased public uptake "
            f"and distribution, {target} might potentially experience increased "
            "sales volume or market penetration."
        ),
        "triggerCondition": (
            "When the updated authorization successfully drives "
            "increased public vaccination rates."
        ),
        "conditionalEffect": f"{target} might potentially see increased sales volume.",
        "counterHypothesis": (
            f"If the updated authorization does not significantly increase public "
            f"uptake, {target} sales volume might remain stable or decrease."
        ),
        "missingEvidence": (
            "Evidence is missing regarding the actual sales volume, distribution "
            "logistics, or market share changes resulting from this authorization."
        ),
        "evidenceBoundary": (
            "The FDA action establishes regulatory permission; it is not "
            f"proof of realized commercial returns for {target}."
        ),
    }


def _faa_proposal() -> dict:
    return {
        "target": "Boeing",
        "sourceFactIds": ["faa-grounding", "faa-inspections"],
        "transmission": "REGULATION",
        "direction": "UNCERTAIN",
        "mechanismJustification": (
            "The issuance of an Emergency Airworthiness Directive (AD) and "
            "the subsequent grounding order impose compliance costs, including "
            "engineering modifications, mandatory inspections, and operational "
            "downtime, on the manufacturer."
        ),
        "triggerCondition": (
            "IF the grounding and required inspections lead to significant "
            "mandatory rework or operational delays for the affected fleet."
        ),
        "conditionalEffect": (
            "Boeing might incur increased operational costs related to "
            "compliance and inspection requirements for the affected 171 aircraft."
        ),
        "counterHypothesis": (
            "The grounding and inspections might not result in significant "
            "compliance costs for Boeing if the required inspections are minimal."
        ),
        "missingEvidence": (
            "The facts do not provide data on Boeing's specific costs associated "
            "with compliance, repair, or lost revenue from the grounding."
        ),
        "evidenceBoundary": (
            "The facts establish the regulatory mandate; they do not establish "
            "that Boeing directly paid airline inspection or grounding costs."
        ),
    }


def test_when_conditional_is_not_false_rejected_for_both_fda_issuers() -> None:
    for issuer in ("ModernaTX", "Pfizer"):
        outcome = review_scenario(_fda_proposal(issuer))
        assert not outcome["blockers"]
        assert outcome["triggerIsExplicitlyConditional"] is True
        assert outcome["counterIsExplicitlyConditional"] is True
        assert outcome["semanticEntailmentVerified"] is False


def test_actual_faa_v2_claim_is_quarantined_even_with_boundary_disclaimer() -> None:
    case = cohort()["cases"][0]
    audit = inspect_evidence_boundary(_faa_proposal(), case)
    assert "UNSUPPORTED_OPERATOR_TO_MANUFACTURER_COST_ATTRIBUTION" in (
        audit["blockers"]
    )
    assert audit["semanticAccuracyVerified"] is False


def test_cautious_faa_hypothetical_manufacturer_impact_remains_reviewable() -> None:
    case = cohort()["cases"][0]
    proposal = _faa_proposal()
    proposal["mechanismJustification"] = (
        "The FAA grounded aircraft operated by airlines. This could lead to "
        "separate manufacturer costs if independent evidence established repair "
        "responsibility, but the agency document itself does not establish it."
    )
    audit = inspect_evidence_boundary(proposal, case)
    assert not audit["blockers"]


class GemmaV2Provider(FakeProvider):
    """Replay actual v2-type proposals through the current full pipeline."""

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        answer = json.loads(super().generate_json(
            system=system, user=user, schema=schema,
        ))
        packet = json.loads(user)
        cat = packet["category"]
        if cat == "AVIATION_SAFETY_REGULATION":
            answer["hypotheses"] = [_faa_proposal()]
        elif cat == "PUBLIC_HEALTH_REGULATION":
            answer["hypotheses"] = [
                _fda_proposal("ModernaTX"),
                _fda_proposal("Pfizer"),
            ]
        return json.dumps(answer)


def test_real_v2_full_pipeline_rejects_unattributed_faa_and_keeps_fda() -> None:
    doc = cohort()
    result = cross.study_cohort(doc, GemmaV2Provider())
    assert [len(x["researchCandidates"]) for x in result["cases"]] == [0, 2, 0]
    assert [len(x["rejectedHypotheses"]) for x in result["cases"]] == [1, 0, 0]
    assert result["review"]["negativeControlAbstained"] is True
    assert result["review"]["economicDecisionConflicts"] == 0
    assert result["review"]["semanticAccuracyVerified"] is False
    assert result["cases"][0]["rejectedHypotheses"][0]["reasonCodes"] == [
        "UNSUPPORTED_OPERATOR_TO_MANUFACTURER_COST_ATTRIBUTION"
    ]


def test_saved_v2_reaudit_reclassifies_without_model_calls_or_original_edits() -> None:
    doc = cohort()
    prior = cross.study_cohort(doc, GemmaV2Provider())
    # Simulate saved v2 results from the former validator: FAA had passed,
    # FDA was rejected by the faulty "When" detector.
    faa = prior["cases"][0]
    faa["researchCandidates"].append({
        **_faa_proposal(), "sourceBoundaryReview": {"blockers": []},
    })
    faa["rejectedHypotheses"] = []
    fda = prior["cases"][1]
    fda["rejectedHypotheses"] = [
        {"reasonCodes": ["TRIGGER_UNCONDITIONAL_WITHOUT_DOWNSTREAM_IF"],
         "proposal": p} for p in [_fda_proposal("ModernaTX"),
                                  _fda_proposal("Pfizer")]
    ]
    fda["researchCandidates"] = []
    old = copy.deepcopy(prior)
    audit = cross.audit_saved_research(doc, prior)
    assert prior == old
    assert audit["auditModelCalls"] == 0
    assert len(audit["cases"][1]["recoverableForHumanReview"]) == 2
    assert audit["summary"]["previousCandidatesNowQuarantined"] == 1
    assert audit["cases"][0]["acceptedCandidateReassessment"][0][
        "reassessment"
    ] == "PREVIOUS_CANDIDATE_NOW_QUARANTINED"
    assert audit["summary"]["semanticAccuracyVerified"] is False
