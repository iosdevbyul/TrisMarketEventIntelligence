"""Actual FAA Jan13/Jan25 Gemma4 failures, offline review and richer decision QA."""
from __future__ import annotations

import copy
import json
import sys
from datetime import date
from pathlib import Path

import pytest
from test_historical_official_event_chain import ChainProvider, load
from tris_market_event_intelligence import historical_official_event_chain_cli as chain
from tris_market_event_intelligence.historical_event_chain_explanation_quality import (
    audit_explanation,
    audit_sequence,
)

COHORT = Path("research/faa-7379max-january-2024-event-chain.json")


def _existing_v1() -> dict:
    """Replay real v1-type inadequacies without invoking Ollama."""
    doc = load()
    source = chain.run_sequence(
        doc, [date(2024, 1, 13), date(2024, 1, 25)], ChainProvider(),
    )
    source["schemaVersion"] = "historical_official_event_chain_sequence_v1"
    early = source["replaySteps"][0]["assessment"]
    early["whatRemainsUnknown"] = (
        "The visible sources do not contain information regarding intentions, "
        "investigations, or necessary future evidence beyond what was announced."
    )
    late = source["replaySteps"][1]["assessment"]
    late["whatChanged"] = (
        "FAA approved an inspection and maintenance procedure for 737-9 MAX "
        "aircraft and decided not to allow production expansion."
    )
    assert len(late["acceptedEdgeExplanations"]) == 3
    for row in late["acceptedEdgeExplanations"]:
        edge = next(
            e for e in doc["curatedEdges"] if e["edgeId"] == row["edgeId"]
        )
        row["missingEvidence"] = edge["basis"]
    return source


def test_actual_v1_replay_flags_unresolved_early_questions() -> None:
    doc = load()
    saved = _existing_v1()
    old = copy.deepcopy(saved)
    report = audit_sequence(doc, saved)
    assert saved == old
    assert report["auditModelCalls"] == 0
    early = report["replayStepAudits"][0]["qualityAudit"]
    assert "UNKNOWN_NARRATIVE_MINIMIZES_OPEN_REGULATORY_QUESTIONS" in (
        early["issueCodes"]
    )
    assert "UNRESOLVED_DECISIONS_LACK_FOLLOWUP_QUESTIONS" in early["issueCodes"]
    assert early["pendingDecisionClaims"]
    assert early["unansweredPendingDecisionClaims"]
    assert report["originalSavedArtifactUnchanged"] is True
    assert report["historicalPITVerified"] is False


def test_actual_v1_late_step_detects_missed_900er_and_empty_missing_evidence() -> None:
    report = audit_sequence(load(), _existing_v1())
    later = report["replayStepAudits"][1]["qualityAudit"]
    assert "INCOMPLETE_NEW_SOURCE_CHANGE_COVERAGE" in later["issueCodes"]
    assert later["missingRecentSourceIds"] == [
        "faa-20240121-900er-inspection", "faa-20240124-production",
    ]
    assert "LINK_EVIDENCE_REQUIREMENTS_NEED_REVIEW" in later["issueCodes"]
    assert len(later["edgeEvidenceReviews"]) == 3
    assert all(
        "MISSING_EVIDENCE_RESTATES_KNOWN_EDGE_BASIS" in r["issueCodes"]
        for r in later["edgeEvidenceReviews"]
    )


def test_model_supplied_changes_and_questions_are_checked_against_visible_ids() -> None:
    packet = chain.prepare_replay(load(), date(2024, 1, 13))
    assessed = {
        "acceptedEdgeExplanations": [],
        "whatRemainsUnknown": (
            "The FAA production audit remains incomplete and the approval "
            "of the full return-to-service process remains undecided."
        ),
        "changeItems": [
            {
                "sourceId": "faa-20240112-oversight",
                "claimId": "production-audit",
                "changedAction": "FAA announced a production-line audit.",
                "scope": "Boeing 737-9 MAX production procedures",
            },
            {
                "sourceId": "faa-20240124-production",
                "claimId": "return-to-service",
                "changedAction": "A future approval must not enter this window.",
                "scope": "A later aircraft approval announcement",
            },
        ],
        "unresolvedQuestions": [
            {
                "sourceId": "faa-20240112-inspection-process",
                "claimId": "additional-data",
                "question": "When would FAA approve the process after the data?",
                "evidenceToResolve": (
                    "A subsequent release confirming a final approval status."
                ),
            },
            {
                "sourceId": "faa-20240124-production",
                "claimId": "return-to-service",
                "question": "Has the later date already approved this procedure?",
                "evidenceToResolve": "Official authorization records.",
            },
        ],
    }
    audit = audit_explanation(packet, assessed)
    assert len(audit["validChangeItems"]) == 1
    assert len(audit["invalidChangeItems"]) == 1
    assert len(audit["validUnresolvedQuestions"]) == 1
    assert len(audit["invalidUnresolvedQuestions"]) == 1
    assert audit["semanticAccuracyVerified"] is False


class RichChainProvider(ChainProvider):
    """Include explicit FAA source changes and pending unresolved questions."""

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        packet = json.loads(user)
        result = json.loads(super().generate_json(
            system=system, user=user, schema=schema,
        ))
        assert "changeItems" in schema["properties"]
        assert "unresolvedQuestions" in schema["properties"]
        result["changeItems"] = [
            {
                "sourceId": source["sourceId"],
                "claimId": source["claims"][0]["claimId"],
                "changedAction": (
                    "The FAA issued or described a discrete regulatory action "
                    "that now appears in the visible official news record."
                ),
                "scope": (
                    "This refers specifically to the aircraft model and "
                    "regulatory step named in the cited source."
                ),
            }
            for source in packet["eligibleSources"]
            if source["sourceId"] not in packet["previouslySeenSourceIds"]
        ]
        result["unresolvedQuestions"] = [
            {
                "sourceId": source["sourceId"],
                "claimId": claim["claimId"],
                "question": (
                    "What further agency decision or completed inspection "
                    "would resolve this announced pending regulatory step?"
                ),
                "evidenceToResolve": (
                    "Independent inspection results or a subsequent release "
                    "confirming approval status must be examined."
                ),
            }
            for source in packet["eligibleSources"]
            for claim in source["claims"]
            if claim["claimId"] in {
                "production-audit", "oversight-intensification", "additional-data",
            }
        ]
        return json.dumps(result)


def test_rich_two_step_replay_tracks_new_sources_and_evidence_questions() -> None:
    doc = load()
    provider = RichChainProvider()
    result = chain.run_sequence(
        doc, [date(2024, 1, 13), date(2024, 1, 25)], provider,
    )
    assert provider.calls == 2
    assert result["schemaVersion"] == "historical_official_event_chain_sequence_v2"
    early, late = result["replaySteps"]
    assert late["previouslySeenSourceIds"] == [
        "faa-20240112-inspection-process", "faa-20240112-oversight",
    ]
    assert early["assessment"]["explanationQualityAudit"]["recentSourceCount"] == 2
    assert late["assessment"]["explanationQualityAudit"]["recentSourceCount"] == 2
    assert not late["assessment"]["explanationQualityAudit"]["missingRecentSourceIds"]
    assert len(late["assessment"]["changeItems"]) == 2
    assert result["historicalPITVerified"] is False


def test_saved_cohort_tampering_rejected_before_any_offline_reclassification() -> None:
    doc = load()
    original = _existing_v1()
    forged = copy.deepcopy(original)
    forged["replaySteps"][0]["eligibleSources"][0]["sourceDate"] = "2024-01-24"
    with pytest.raises(ValueError, match="EVENT_CHAIN_SAVED_SOURCE_SCOPE_INVALID"):
        audit_sequence(doc, forged)
    forged = copy.deepcopy(original)
    forged["sourceCohortSha256"] = "wrong"
    with pytest.raises(ValueError, match="EVENT_CHAIN_SAVED_SEQUENCE_INTEGRITY_INVALID"):
        audit_sequence(doc, forged)


def test_offline_existing_cli_never_invokes_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    saved = tmp_path / "saved-v1.json"
    saved.write_text(json.dumps(_existing_v1()), encoding="utf-8")
    output = tmp_path / "review-v1.json"
    monkeypatch.setattr(sys, "argv", [
        "historical-event-chain", "--chain", str(COHORT),
        "--audit-existing", str(saved), "--output", str(output),
    ])
    monkeypatch.setattr(
        chain, "build_runtime",
        lambda *args, **kwargs: pytest.fail("Ollama must not run"),
    )
    chain.main()
    report = json.loads(capsys.readouterr().out)
    assert report["auditModelCalls"] == 0
    assert output.exists()
    with pytest.raises(ValueError, match="Expected new private output"):
        chain.main()
