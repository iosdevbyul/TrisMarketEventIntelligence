"""Post-v1 audit of real MSFT, Alphabet and AMD economic reasoning."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from test_historical_official_fact_reasoning import SourceFactProvider

from tris_market_event_intelligence import historical_official_fact_reasoning_cli as official
from tris_market_event_intelligence.historical_official_mechanism_audit import (
    inspect_interpretation,
    inspect_mechanism,
)

COHORT = Path("research/independent-historical-news-2023.json")


def _cohort() -> dict:
    return json.loads(COHORT.read_text(encoding="utf-8"))


def test_regulation_requires_primary_regulatory_fact_and_matching_mechanism() -> None:
    for case in _cohort()["cases"]:
        facts = {fact["factId"]: fact for fact in case["facts"]}
        proposal = {
            "sourceFactIds": [case["facts"][0]["factId"]],
            "transmission": "REGULATION",
            "triggerCondition": "If the business revenue trend persists",
            "conditionalEffect": "Profitability could increase over time.",
            "counterHypothesis": "If customer demand weakens it may reverse.",
        }
        audit = inspect_mechanism(proposal, facts)
        assert not audit["acceptedForQualitativeReview"]
        assert "REGULATION_CHANNEL_NOT_GROUNDED_IN_SOURCE_OR_MECHANISM" in (
            audit["blockers"]
        )
        assert audit["economicChannelVerified"] is False


def test_made_up_thresholds_are_not_promoted_to_source_grounded_scenarios() -> None:
    case = _cohort()["cases"][1]
    facts = {fact["factId"]: fact for fact in case["facts"]}
    proposal = {
        "sourceFactIds": ["googl-cloud", "googl-operating"],
        "transmission": "REVENUE",
        "triggerCondition": (
            "If revenue growth exceeds 10% and operating income tops 400 million"
        ),
        "conditionalEffect": "The cloud segment could improve margins.",
        "counterHypothesis": (
            "If the operating income stays below 350 million, growth may slow."
        ),
    }
    audit = inspect_mechanism(proposal, facts)
    assert "UNSOURCED_NUMERIC_THRESHOLD" in audit["blockers"]
    assert set(audit["unsupportedNumericValues"]) >= {"10", "400", "350"}
    assert not audit["acceptedForQualitativeReview"]


def test_regulatory_source_and_regulatory_hypothesis_not_automatically_rejected() -> None:
    facts = {"law": {"statement": (
        "The government announced a new compliance regulation for exports."
    )}}
    proposal = {
        "sourceFactIds": ["law"],
        "transmission": "REGULATION",
        "triggerCondition": "If the regulation remains in place",
        "conditionalEffect": "Compliance costs could rise.",
        "counterHypothesis": "If the rule is rescinded, costs might stay flat.",
    }
    audit = inspect_mechanism(proposal, facts)
    assert not audit["blockers"]
    assert audit["acceptedForQualitativeReview"]
    assert audit["semanticEntailmentVerified"] is False
    assert audit["economicChannelVerified"] is False


def test_unsupported_high_margin_descriptor_is_flagged_without_rewriting() -> None:
    source = "Microsoft Cloud quarterly revenue reached USD 30.3 billion."
    explanation = "This highlights the rapid growth of its high-margin cloud services."
    assert inspect_interpretation(explanation, source) == [
        "UNSUPPORTED_EVALUATIVE_QUALIFIER"
    ]


class MisclassifiedProvider(SourceFactProvider):
    """Reproduce live error: every financial mechanism labeled REGULATION."""

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        response = json.loads(super().generate_json(
            system=system, user=user, schema=schema,
        ))
        response["mechanisms"][0]["transmission"] = "REGULATION"
        return json.dumps(response)


def test_three_actual_case_outputs_quarantined_without_relabeling() -> None:
    report = official.run_cohort(_cohort(), MisclassifiedProvider())
    assert report["modelCalls"] == 3
    assert all(not case["economicMechanisms"] for case in report["cases"])
    for case in report["cases"]:
        bad = case["rejectedEconomicMechanisms"]
        assert len(bad) == 1
        assert bad[0]["proposal"]["transmission"] == "REGULATION"
        assert bad[0]["reason"] == "ECONOMIC_MECHANISM_QUALITY_GATE"
        assert "REGULATION_CHANNEL_NOT_GROUNDED_IN_SOURCE_OR_MECHANISM" in (
            bad[0]["qualityAudit"]["blockers"]
        )
    assert all(
        count == 1 for count in
        report["crossEventReview"]["qualityGateRejectionCounts"].values()
    )


def test_existing_three_case_artifact_reaudited_without_model_reexecution() -> None:
    cohort = _cohort()
    old = official.run_cohort(cohort, SourceFactProvider())
    # Emulate an existing legacy v1 artifact that originally accepted the
    # wrong economic labels. Leave model content untouched in original.
    for case in old["cases"]:
        for mechanism in case["economicMechanisms"]:
            mechanism["transmission"] = "REGULATION"
            mechanism.pop("qualityAudit")
    original = copy.deepcopy(old)
    audit = official.audit_existing(cohort, old)
    assert old == original
    assert audit["auditModelCalls"] == 0
    assert audit["originalModelCalls"] == 3
    assert audit["schemaVersion"] == "historical_official_fact_reasoning_audit_v1"
    assert sum(len(x["economicMechanisms"]) for x in audit["cases"]) == 0
    assert sum(len(x["rejectedEconomicMechanisms"]) for x in audit["cases"]) == 3
    assert audit["crossEventReview"]["generalizationVerified"] is False
    assert audit["trainingPerformed"] is False
    assert audit["tradesPlaced"] is False


def test_original_cohort_sha_and_fact_tampering_refused() -> None:
    cohort = _cohort()
    report = official.run_cohort(cohort, SourceFactProvider())
    report["sourceCohortSha256"] = "f" * 64
    with pytest.raises(ValueError, match="INTEGRITY_INVALID"):
        official.audit_existing(cohort, report)
    report = official.run_cohort(cohort, SourceFactProvider())
    report["cases"][0]["factInterpretations"][0]["originalSourceFact"] = (
        "Invented earnings"
    )
    with pytest.raises(ValueError, match="FACT_TAMPERED"):
        official.audit_existing(cohort, report)
