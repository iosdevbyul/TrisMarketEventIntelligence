"""FDA temporal contradictions and evidence ledger without replaying Ollama."""
from __future__ import annotations

import copy
import json
import sys
from datetime import date
from pathlib import Path

import pytest
from test_historical_cross_event_validation import fda

from tris_market_event_intelligence import historical_official_event_chain_cli as chain
from tris_market_event_intelligence.historical_temporal_question_lifecycle import (
    audit_saved_temporal,
    audit_temporal_consistency,
    evidence_ledger,
    question_snapshot,
    validate_question_tracks,
)

TRACKS = Path("research/fda-2023-question-lifecycle.json")
FDA = Path("research/fda-covid-vaccine-2023-event-chain.json")


def tracks() -> dict:
    return json.loads(TRACKS.read_text(encoding="utf-8"))


class MinimalProvider:
    model_id = "mock-fda-temporal"

    def __init__(self) -> None:
        self.calls = 0

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        self.calls += 1
        packet = json.loads(user)
        assert "questionTracks" not in packet
        assert "questionLifecycle" not in packet
        assert "futureQuestions" not in packet
        assert "eligibleSources" in packet
        answer = {
            "factInterpretations": [
                {
                    "sourceId": source["sourceId"],
                    "claimId": c["claimId"],
                    "interpretation": (
                        "The FDA announced this regulatory statement with "
                        "a specific named product or action and its scope."
                    ),
                    "limitation": (
                        "This release alone does not document completed "
                        "commercial sales or independent economic causality."
                    ),
                }
                for source in packet["eligibleSources"]
                for c in source["claims"]
            ],
            "edgeExplanations": [
                {
                    "edgeId": edge["edgeId"],
                    "fromSourceId": edge["fromSourceId"],
                    "toSourceId": edge["toSourceId"],
                    "interpretation": (
                        "This dated source follows the other official statement "
                        "but does not prove a causal chain of commercial effects."
                    ),
                    "missingEvidence": (
                        "Independent agency review records would be needed "
                        "for proof of the specific regulatory mechanism."
                    ),
                }
                for edge in packet["visibleCuratedEdges"]
            ],
            "whatChanged": (
                "The FDA moved from a formula recommendation toward product "
                "specific authorizations across separate manufacturers."
            ),
            "whatRemainsUnknown": (
                "The manufacturers' underlying detailed filing receipts "
                "and full FDA review documents are not in the source packet."
            ),
            "changeItems": [],
            "unresolvedQuestions": [],
        }
        return json.dumps(answer)


def test_fda_annotations_do_not_change_original_source_bytes_or_sha() -> None:
    doc = fda()
    annotations = tracks()
    chain.validate_chain(doc)
    validate_question_tracks(doc, annotations)
    assert "questionTracks" not in doc
    assert len(annotations["tracks"]) == 2
    assert annotations["eventId"] == doc["eventId"]


def test_question_states_evolve_without_future_source_leak() -> None:
    doc, annotations = fda(), tracks()
    april = question_snapshot(
        annotations, chain.prepare_replay(doc, date(2023, 4, 19)),
    )
    june = question_snapshot(
        annotations, chain.prepare_replay(doc, date(2023, 6, 17)),
    )
    september = question_snapshot(
        annotations, chain.prepare_replay(doc, date(2023, 9, 12)),
    )
    assert len(april) == 1
    assert april[0]["status"] == "OPEN"
    assert len(june) == 2
    assert {x["status"] for x in june} == {"OPEN", "PARTIALLY_RESOLVED"}
    assert {x["status"] for x in september} == {
        "RESOLVED_BY_LATER_SOURCE", "PARTIALLY_RESOLVED",
    }
    assert "fda-20230911-mrna" not in {
        q["supportingSourceId"] for q in june
    }
    assert all(not q["semanticAccuracyVerified"] for q in september)


def test_deterministic_ledger_recovers_changes_even_when_model_emits_empty_array() -> None:
    doc = fda()
    packet = chain.prepare_replay(doc, date(2023, 10, 4))
    packet["previouslySeenSourceIds"] = [
        "fda-20230418-bivalent", "fda-20230616-formula",
    ]
    ledger = evidence_ledger(packet)
    assert len(ledger) == 8
    assert [row["sourceId"] for row in ledger if row["newSincePreviousCutoff"]] == [
        "fda-20230911-mrna", "fda-20230911-mrna",
        "fda-20231003-novavax", "fda-20231003-novavax",
    ]
    assert all(row["origin"] == "CURATOR_SOURCE_CLAIM_NOT_MODEL_INFERENCE"
               for row in ledger)
    audit = audit_temporal_consistency(packet, {
        "changeItems": [], "unresolvedQuestions": [],
        "whatRemainsUnknown": "The filing receipts remain unavailable.",
    }, question_snapshot(tracks(), packet))
    assert audit["modelMissingChangeSourceIds"] == [
        "fda-20230911-mrna", "fda-20231003-novavax",
    ]
    assert audit["curatedChangeLedgerCount"] == 8


def test_actual_fda_late_text_flags_resolved_formula_but_preserves_filing_unknown() -> None:
    doc, annotations = fda(), tracks()
    provider = MinimalProvider()
    saved = chain.run_sequence(
        doc, [date(2023, 6, 17), date(2023, 10, 4)], provider,
    )
    before = copy.deepcopy(saved)
    late = saved["replaySteps"][1]["assessment"]
    late["whatRemainsUnknown"] = (
        "The specific composition of the fall 2023 strain discussed in June "
        "remains an unobserved decision. Furthermore the outcome of anticipated "
        "data submissions remains unknown."
    )
    late["unresolvedQuestions"] = [
        {
            "sourceId": "fda-20230418-bivalent",
            "claimId": "fall-formula-pending",
            "question": "What is the final composition of the fall 2023 strain?",
            "evidenceToResolve": (
                "A final official decision document discussing the formulation."
            ),
        },
        {
            "sourceId": "fda-20230616-formula",
            "claimId": "filing-still-needed",
            "question": "Were manufacturer data receipts actually recorded?",
            "evidenceToResolve": (
                "A source record of the receipt of supporting datasets."
            ),
        },
    ]
    before = copy.deepcopy(saved)
    report = audit_saved_temporal(doc, saved, annotations)
    assert saved == before
    assert provider.calls == 2
    assert report["auditModelCalls"] == 0
    assert report["originalSourceCohortUnchanged"] is True
    assert report["originalSavedArtifactUnchanged"] is True
    early, latest = report["replayStepAudits"]
    assert len(early["questionLifecycle"]) == 2
    assert latest["modelTemporalAudit"][
        "potentialNarrativeFormulaContradiction"
    ] is True
    assert latest["modelTemporalAudit"]["staleQuestionAnchors"][0][
        "questionId"
    ] == "fda-2023-xbb15-authorized-formula"
    assert not any(
        x["questionId"] == "fda-2023-manufacturer-filings-evidence"
        for x in latest["modelTemporalAudit"]["staleQuestionAnchors"]
    )
    assert latest["questionLifecycleDelta"][0]["previousStatus"] == (
        "PARTIALLY_RESOLVED"
    )
    assert latest["questionLifecycleDelta"][0]["currentStatus"] == (
        "RESOLVED_BY_LATER_SOURCE"
    )


def test_bad_source_anchor_and_backward_transition_rejected() -> None:
    doc = fda()
    forged = copy.deepcopy(tracks())
    forged["tracks"][0]["transitions"][2]["claimId"] = "invented"
    with pytest.raises(ValueError, match="QUESTION_TRANSITION_SOURCE_INVALID"):
        validate_question_tracks(doc, forged)
    forged = copy.deepcopy(tracks())
    forged["tracks"][0]["transitions"][2]["status"] = "OPEN"
    with pytest.raises(ValueError, match="QUESTION_TRANSITION_INVALID"):
        validate_question_tracks(doc, forged)
    forged = copy.deepcopy(tracks())
    forged["eventId"] = "WRONG"
    with pytest.raises(ValueError, match="QUESTION_TRACK_ARTIFACT_INVALID"):
        validate_question_tracks(doc, forged)


def test_new_two_cutoff_model_does_not_receive_later_editorial_knowledge() -> None:
    doc, annotations = fda(), tracks()
    provider = MinimalProvider()
    saved = chain.run_sequence(
        doc, [date(2023, 6, 17), date(2023, 10, 4)],
        provider, question_tracks=annotations,
    )
    assert provider.calls == 2
    assert saved["replaySteps"][0]["questionLifecycle"][0]["status"] == (
        "PARTIALLY_RESOLVED"
    )
    assert saved["replaySteps"][1]["questionLifecycle"][0]["status"] == (
        "RESOLVED_BY_LATER_SOURCE"
    )
    assert len(saved["questionLifecycleDelta"]) == 1
    assert "MODEL_CHANGE_ITEMS_OMIT_VISIBLE_NEW_SOURCES" in (
        saved["replaySteps"][1]["assessment"][
            "temporalConsistencyAudit"
        ]["issueCodes"]
    )
    assert saved["semanticAccuracyVerified"] is False


def test_existing_cli_with_annotations_is_zero_model_and_create_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    doc = fda()
    model = MinimalProvider()
    old = chain.run_sequence(
        doc, [date(2023, 6, 17), date(2023, 10, 4)], model,
    )
    saved = tmp_path / "existing-fda-v2.json"
    saved.write_text(json.dumps(old), encoding="utf-8")
    result = tmp_path / "new-fda-audit.json"
    monkeypatch.setattr(sys, "argv", [
        "historical-event-chain",
        "--chain", str(FDA),
        "--question-tracks", str(TRACKS),
        "--audit-existing", str(saved),
        "--output", str(result),
    ])
    monkeypatch.setattr(
        chain, "build_runtime",
        lambda *args, **kwargs: pytest.fail("must never invoke model"),
    )
    chain.main()
    summary = json.loads(capsys.readouterr().out)
    assert summary["auditModelCalls"] == 0
    assert summary["questionTrackCount"] == 2
    output = json.loads(result.read_text(encoding="utf-8"))
    assert output["temporalQuestionLifecycleAudit"]["auditModelCalls"] == 0
    with pytest.raises(ValueError, match="Expected new private output"):
        chain.main()
