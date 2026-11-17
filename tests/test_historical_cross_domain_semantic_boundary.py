"""Real agency error-shape regressions from observed Mac Gemma4 audit."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from test_historical_cross_domain_official_study import FakeProvider, cohort

from tris_market_event_intelligence import historical_cross_domain_official_study_cli as cross
from tris_market_event_intelligence.historical_cross_domain_semantic_boundary import (
    describe_abstention_format,
    inspect_evidence_boundary,
)
from tris_market_event_intelligence.historical_official_mechanism_audit import (
    inspect_mechanism,
)

COHORT = Path("research/cross-domain-official-events-2023-2024.json")


def _proposal(case: dict, *, transmission: str, target: str,
              justification: str) -> dict:
    return {
        "target": target,
        "sourceFactIds": [row["factId"] for row in case["facts"][-2:]],
        "transmission": transmission,
        "direction": "UNCERTAIN",
        "mechanismJustification": justification,
        "triggerCondition": "If the official conditions continue to apply",
        "conditionalEffect": (
            "The commercial outlook could change if other independent "
            "economic conditions support the transmission mechanism."
        ),
        "counterHypothesis": (
            "If business exposure is small, financial impact might be minimal."
        ),
        "missingEvidence": (
            "Additional issuer financial and operational facts are required "
            "before inferring a realized outcome."
        ),
        "evidenceBoundary": (
            "The official action is established, but company economic "
            "results are not established by the agency release alone."
        ),
    }


def test_faa_airline_inspections_do_not_directly_prove_boeing_expenses() -> None:
    faa = cohort()["cases"][0]
    actual_justification = (
        "The mandatory grounding and airworthiness directive impose "
        "operational restrictions and compliance costs on Boeing by requiring "
        "immediate corrective actions, maintenance, and operational adjustments "
        "for the affected fleet."
    )
    proposal = _proposal(
        faa, transmission="REGULATION", target="Boeing",
        justification=actual_justification,
    )
    audit = inspect_evidence_boundary(proposal, faa)
    assert "UNSUPPORTED_OPERATOR_TO_MANUFACTURER_COST_ATTRIBUTION" in (
        audit["blockers"]
    )
    # A cautious conditional chain remains reviewable; the source can
    # establish operators' inspections, not Boeing's accounting incidence.
    proposal["mechanismJustification"] = (
        "The FAA grounded certain aircraft operated by airlines; this might "
        "later affect Boeing through customer, delivery or remediation channels."
    )
    assert not inspect_evidence_boundary(proposal, faa)["blockers"]


def test_fda_authorization_is_real_regulatory_evidence_not_a_sale() -> None:
    fda = cohort()["cases"][1]
    proposal = _proposal(
        fda, transmission="REGULATION", target="ModernaTX",
        justification=(
            "The FDA's authorization for emergency use enables ModernaTX "
            "to distribute the updated vaccine through a regulatory pathway."
        ),
    )
    proposal["sourceFactIds"] = ["fda-authorization", "fda-previous"]
    facts = {x["factId"]: x for x in fda["facts"]}
    audit = inspect_mechanism(proposal, facts)
    assert "REGULATION_CHANNEL_NOT_GROUNDED_IN_SOURCE_OR_MECHANISM" not in (
        audit["blockers"]
    )
    boundary = inspect_evidence_boundary(proposal, fda)
    assert "AUTHORIZATION_IS_NOT_EVIDENCE_OF_REALIZED_SALES" in (
        boundary["warnings"]
    )
    assert boundary["semanticAccuracyVerified"] is False


def test_issuer_cloud_revenue_does_not_become_a_regulatory_event() -> None:
    facts = {"cloud": {"statement": (
        "Reported cloud revenue increased during the quarter."
    )}}
    proposal = _proposal(
        {"facts": [{"factId": "cloud"}, {"factId": "cloud"}]},
        transmission="REGULATION", target="CloudCo",
        justification="Cloud company authorization leads to higher revenue.",
    )
    proposal["sourceFactIds"] = ["cloud"]
    audit = inspect_mechanism(proposal, facts)
    assert "REGULATION_CHANNEL_NOT_GROUNDED_IN_SOURCE_OR_MECHANISM" in (
        audit["blockers"]
    )


def test_nasa_sentinel_is_attempted_abstention_but_format_failure() -> None:
    nasa = cohort()["cases"][2]
    source = _proposal(
        nasa, transmission="REGULATION", target="None",
        justification=(
            "No justified commercial target was identified. The NASA sample "
            "return is a scientific event and not evidence of regulation."
        ),
    )
    result = describe_abstention_format(nasa, [source])
    assert result["modelExpressedNoTargetViaSentinel"] is True
    assert result["abstentionOutputFormatViolation"] is True
    assert result["negativeControlPass"] is False
    assert result["modelAbstainedWithEmptyHypotheses"] is False
    assert describe_abstention_format(nasa, [])["negativeControlPass"] is True


class SentinelProvider(FakeProvider):
    """Simulate v1's None sentinel encoded as a fake NASA hypothesis."""

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        payload = json.loads(super().generate_json(
            system=system, user=user, schema=schema,
        ))
        packet = json.loads(user)
        if packet["category"] == "SCIENTIFIC_MISSION_NEGATIVE_CONTROL":
            payload["economicDecision"] = "ABSTAIN"
            payload["hypotheses"] = [_proposal(
                cohort()["cases"][2], transmission="REGULATION", target="None",
                justification=(
                    "No justified economic target exists for NASA sample "
                    "return; it is scientific rather than a regulatory action."
                ),
            )]
        return json.dumps(payload)


def test_new_model_records_conflicting_abstention_without_promotion() -> None:
    nasa = cohort()["cases"][2]
    result = cross.study_case(nasa, SentinelProvider())
    assert result["economicDecision"] == "ABSTAIN"
    assert result["economicDecisionConsistent"] is False
    assert result["negativeControlAbstention"] is False
    assert result["abstentionFormatAudit"]["modelExpressedNoTargetViaSentinel"]
    assert result["researchCandidates"] == []
    assert result["rejectedHypotheses"][0]["reasonCodes"] == [
        "ECONOMIC_DECISION_CONTRADICTS_HYPOTHESES"
    ]


def test_legacy_v1_saved_file_sentinel_diagnosed_without_model_calls() -> None:
    doc = cohort()
    prior = cross.study_cohort(doc, SentinelProvider())
    prior["schemaVersion"] = cross.LEGACY_RESULT_VERSION
    previous = copy.deepcopy(prior)
    reviewed = cross.audit_saved_research(doc, prior)
    assert previous == prior
    assert reviewed["auditModelCalls"] == 0
    assert reviewed["summary"]["negativeControlFormatViolations"] == 1
    assert reviewed["summary"]["negativeControlModelAbstained"] is False
    nasa = reviewed["cases"][2]
    assert nasa["abstentionFormatAudit"]["modelExpressedNoTargetViaSentinel"]
    assert nasa["rejectionDiagnostics"][0]["classification"] == (
        "NEGATIVE_CONTROL_MODEL_ABSTENTION_FAILURE"
    )
    assert nasa["recoverableForHumanReview"] == []
    assert reviewed["summary"]["semanticAccuracyVerified"] is False


def test_fda_commercial_guarantee_still_blocked() -> None:
    fda = cohort()["cases"][1]
    proposal = _proposal(
        fda, target="Pfizer", transmission="REGULATION",
        justification=(
            "The FDA approved an updated vaccine, which will generate "
            "profit for the issuer even without evidence of demand."
        ),
    )
    review = inspect_evidence_boundary(proposal, fda)
    assert "UNDOCUMENTED_COMMERCIAL_OUTCOME_ASSERTED" in review["blockers"]
