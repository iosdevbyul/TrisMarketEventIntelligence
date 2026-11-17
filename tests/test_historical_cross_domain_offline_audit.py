"""Offline re-audit of saved real cross-domain research without rerunning LLM."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
from test_historical_cross_domain_official_study import FakeProvider, cohort

from tris_market_event_intelligence import historical_cross_domain_official_study_cli as cross


def test_saved_research_reaudit_retains_original_and_science_failure() -> None:
    source = cohort()
    model = FakeProvider(invent_target=True)
    original = cross.study_cohort(source, model)
    assert original["review"]["negativeControlAbstained"] is False
    assert len(original["cases"][2]["rejectedHypotheses"]) == 1
    before = copy.deepcopy(original)
    result = cross.audit_saved_research(source, original)
    assert original == before
    assert result["auditModelCalls"] == 0
    assert result["originalResearchModelCalls"] == 3
    assert result["summary"]["negativeControlModelAbstained"] is False
    assert result["summary"]["originalRejectedHypotheses"] == 2
    assert result["cases"][2]["modelHypothesesProposed"] == 1
    assert result["cases"][2]["rejectionDiagnostics"][0]["classification"] == (
        "NEGATIVE_CONTROL_MODEL_ABSTENTION_FAILURE"
    )
    assert result["cases"][2]["recoverableForHumanReview"] == []
    assert result["promotionAllowed"] is False
    assert result["trainingPerformed"] is False


def test_saved_research_source_sha_and_fact_provenance_required() -> None:
    doc = cohort()
    prior = cross.study_cohort(doc, FakeProvider())
    prior["sourceCohortSha256"] = "wrong"
    with pytest.raises(ValueError, match="INTEGRITY_INVALID"):
        cross.audit_saved_research(doc, prior)
    prior = cross.study_cohort(doc, FakeProvider())
    prior["cases"][0]["factAssessments"][0]["sourceStatement"] = "Faked text"
    with pytest.raises(ValueError, match="FACT_PROVENANCE_INVALID"):
        cross.audit_saved_research(doc, prior)
    prior = cross.study_cohort(doc, FakeProvider())
    prior["cases"][2]["negativeControlAbstention"] = False
    with pytest.raises(ValueError, match="ABSTENTION_INTEGRITY_INVALID"):
        cross.audit_saved_research(doc, prior)


def test_no_gemma_calls_for_occupied_destination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    occupied = tmp_path / "existing.json"
    occupied.write_text('{"valid":"original"}', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "historical-cross-domain", "--cohort",
        "research/cross-domain-official-events-2023-2024.json",
        "--run", "--output", str(occupied),
    ])
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("LLM provider must not be initialized")
    monkeypatch.setattr(cross, "build_runtime", forbidden)
    with pytest.raises(ValueError, match="Expected new private output"):
        cross.main()
    assert occupied.read_text(encoding="utf-8") == '{"valid":"original"}'


def test_rejected_saved_hypothesis_is_classified_from_current_quality() -> None:
    doc = cohort()
    old = cross.study_cohort(doc, FakeProvider())
    proposal = {
        **old["cases"][0]["researchCandidates"][0],
        "target": "Boeing",
    }
    original = {
        key: proposal[key] for key in cross.Explanation.model_fields
    }
    old["cases"][0]["researchCandidates"] = []
    old["cases"][0]["rejectedHypotheses"].append({
        "reasonCodes": ["TRIGGER_NOT_CONDITIONAL"],
        "proposal": original,
        "scenarioReview": None,
        "economicAudit": None,
    })
    audited = cross.audit_saved_research(doc, old)
    entry = audited["cases"][0]["rejectionDiagnostics"][0]
    assert entry["classification"] == "ELIGIBLE_FOR_HUMAN_REVIEW_ONLY"
    assert len(audited["cases"][0]["recoverableForHumanReview"]) == 1
    assert entry["semanticAccuracyVerified"] is False
    assert audited["summary"]["generalizationVerified"] is False
