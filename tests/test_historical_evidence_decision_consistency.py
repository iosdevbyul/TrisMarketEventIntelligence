"""The actual FDA v2 model mistakes must remain visible, not be reclassified as success."""
from __future__ import annotations

import copy
import json
from datetime import date

import pytest
from test_historical_cross_event_validation import fda
from test_historical_staged_reasoning import StageProvider
from test_historical_temporal_question_lifecycle import tracks

from tris_market_event_intelligence import historical_official_event_chain_cli as chain
from tris_market_event_intelligence.historical_evidence_decision_consistency import (
    check_action_evidence,
    review_evidence_decisions,
)
from tris_market_event_intelligence.historical_staged_reasoning import (
    StagedDraftEvidenceError,
    produce_draft,
    visible_questions,
)
from tris_market_event_intelligence.historical_temporal_question_lifecycle import (
    audit_saved_temporal,
)


def packet(cutoff: date) -> dict:
    value = chain.prepare_replay(fda(), cutoff)
    value["previouslySeenSourceIds"] = []
    return value


def test_real_june_fda_status_contradicts_its_own_missing_records_explanation() -> None:
    evidence = packet(date(2023, 6, 17))
    result = review_evidence_decisions(evidence, {
        "visibleQuestions": visible_questions(tracks(), evidence),
        "modelAuthoredChangeItems": [
            {"sourceId": "fda-20230418-bivalent",
             "claimId": "fall-formula-pending",
             "changedAction": "Included in visible historical evidence."},
            {"sourceId": "fda-20230616-formula",
             "claimId": "xbb-recommendation",
             "changedAction": "Included in visible historical evidence."},
        ],
        "modelAuthoredQuestionAssessments": [
            {
                "questionId": "fda-2023-xbb15-authorized-formula",
                "status": "PARTIALLY_RESOLVED",
                "sourceId": "fda-20230616-formula",
                "claimId": "xbb-recommendation",
                "explanation": (
                    "The FDA recommended XBB.1.5, but no product was "
                    "formally authorized at this historical cutoff."
                ),
                "stillUnknown": "The final product authorization.",
            },
            {
                "questionId": "fda-2023-manufacturer-filings-evidence",
                "status": "RESOLVED_BY_VISIBLE_SOURCE",
                "sourceId": "fda-20230616-formula",
                "claimId": "filing-still-needed",
                "explanation": (
                    "The visible FDA source confirms that the records "
                    "of receipt and full review are not yet available."
                ),
                "stillUnknown": (
                    "The actual records establishing receipt and "
                    "review are not visible."
                ),
            },
        ],
    })
    assert result["issueCodes"] == [
        "VACUOUS_MODEL_CHANGE_ACTION",
        "MODEL_STATUS_CONTRADICTS_OWN_EVIDENCE_EXPLANATION",
    ]
    assert len(result["vacuousChangeItems"]) == 2
    assert result["contradictoryDecisions"][0]["questionId"] == (
        "fda-2023-manufacturer-filings-evidence"
    )
    assert result["modelOutputRewritten"] is False
    assert result["semanticAccuracyVerified"] is False


def test_later_product_authorization_may_have_unresolved_detailed_records() -> None:
    evidence = packet(date(2023, 10, 4))
    draft = {
        "visibleQuestions": visible_questions(tracks(), evidence),
        "modelAuthoredChangeItems": [{
            "sourceId": "fda-20230911-mrna",
            "claimId": "mrna-authorization",
            "changedAction": (
                "FDA approved and authorized the updated monovalent "
                "XBB.1.5 ModernaTX and Pfizer mRNA vaccines."
            ),
        }],
        "modelAuthoredQuestionAssessments": [{
            "questionId": "fda-2023-xbb15-authorized-formula",
            "status": "RESOLVED_BY_VISIBLE_SOURCE",
            "sourceId": "fda-20230911-mrna",
            "claimId": "mrna-authorization",
            "explanation": (
                "FDA approved and authorized updated monovalent "
                "XBB.1.5 mRNA vaccines from ModernaTX and Pfizer."
            ),
            "stillUnknown": (
                "Underlying internal review documents and manufacturer "
                "filing details are not visible."
            ),
        }],
    }
    check = review_evidence_decisions(evidence, draft)
    assert check["issueCodes"] == []
    assert check["contradictoryDecisions"] == []


def test_stage_one_rejects_vacuous_action_without_synthesizing_text() -> None:
    class VacuousProvider(StageProvider):
        def generate_json(self, *, system: str, user: str, schema: dict) -> str:
            raw = json.loads(super().generate_json(
                system=system, user=user, schema=schema,
            ))
            if "questionAssessments" in schema["properties"]:
                for item in raw["changeItems"]:
                    item["changedAction"] = "Included in visible historical evidence."
            return json.dumps(raw)

    evidence = packet(date(2023, 6, 17))
    provider = VacuousProvider()
    with pytest.raises(StagedDraftEvidenceError) as failure:
        produce_draft(provider, evidence, tracks())
    assert provider.calls == 1
    assert "VACUOUS_MODEL_CHANGE_ACTION" in failure.value.diagnostic["reasonCodes"]
    assert len(failure.value.diagnostic["invalidActionItems"]) == 2
    assert failure.value.diagnostic["curatorFactsNotSubstituted"] is True
    assert check_action_evidence(evidence, [{
        "sourceId": "fda-20230616-formula",
        "claimId": "xbb-recommendation",
        "changedAction": "Included in visible historical evidence.",
    }])


def test_saved_staged_review_reconstructs_visible_questions_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc, annotations = fda(), tracks()
    provider = StageProvider()
    original = chain.run_sequence(
        doc, [date(2023, 6, 17), date(2023, 10, 4)],
        provider, question_tracks=annotations, reasoning_mode="staged",
    )
    before = copy.deepcopy(original)
    early = original["replaySteps"][0]["assessment"]["stageOneModelDraft"]
    early["modelAuthoredChangeItems"][0]["changedAction"] = (
        "Included in visible historical evidence."
    )
    early["modelAuthoredQuestionAssessments"][1]["status"] = (
        "RESOLVED_BY_VISIBLE_SOURCE"
    )
    early["modelAuthoredQuestionAssessments"][1]["explanation"] = (
        "FDA records establishing receipt and review are not yet available."
    )
    old = copy.deepcopy(original)
    monkeypatch.setattr(
        "tris_market_event_intelligence.historical_staged_reasoning.produce_draft",
        lambda *_args, **_kwargs: pytest.fail("No model allowed"),
    )
    report = audit_saved_temporal(doc, original, annotations)
    assert original == old
    assert len(report["replayStepAudits"]) == 2
    early_review = report["replayStepAudits"][0][
        "savedEvidenceDecisionConsistencyReview"
    ]
    assert early_review["issueCodes"] == [
        "VACUOUS_MODEL_CHANGE_ACTION",
        "MODEL_STATUS_CONTRADICTS_OWN_EVIDENCE_EXPLANATION",
    ]
    assert report["replayStepAudits"][1][
        "savedEvidenceDecisionConsistencyReview"
    ]["issueCodes"] == []
    assert report["auditModelCalls"] == 0
    assert before["replaySteps"][0]["assessment"]["stageOneModelDraft"][
        "modelAuthoredChangeItems"
    ][0]["changedAction"] != "Included in visible historical evidence."
