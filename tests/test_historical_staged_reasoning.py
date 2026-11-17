"""Staged generation: real FDA input and old failure modes, no Ollama needed."""
from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path

import pytest
from test_historical_cross_event_validation import fda
from test_historical_temporal_question_lifecycle import tracks

from tris_market_event_intelligence import historical_official_event_chain_cli as chain
from tris_market_event_intelligence.historical_staged_reasoning import (
    EvidenceFirstDraft,
    produce_draft,
    visible_questions,
)


class StageProvider:
    model_id = "mock-staged"

    def __init__(
        self, *, skip_source: bool = False, invent_future: bool = False,
        omit_question: bool = False, unresolved_everything: bool = False,
    ) -> None:
        self.calls = 0
        self.inputs: list[dict] = []
        self.skip_source = skip_source
        self.invent_future = invent_future
        self.omit_question = omit_question
        self.unresolved_everything = unresolved_everything

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        self.calls += 1
        envelope = json.loads(user)
        self.inputs.append(envelope)
        if "changeItems" in schema["properties"] and (
            "questionAssessments" in schema["properties"]
        ):
            packet = envelope["visibleHistoricalEvidence"]
            visible = packet["eligibleSources"]
            fresh = [
                source for source in visible
                if source["sourceId"] not in packet["previouslySeenSourceIds"]
            ]
            if self.skip_source:
                fresh = fresh[:-1]
            changes = [
                {
                    "sourceId": source["sourceId"],
                    "claimId": source["claims"][0]["claimId"],
                    "changedAction": (
                        "The regulator issued this updated statement which "
                        "changed the pending formal product approval position."
                    ),
                    "scope": (
                        "The specific product and agency regulatory decision "
                        "described by this official dated release."
                    ),
                }
                for source in fresh
            ]
            if self.invent_future:
                changes[0]["sourceId"] = "fda-20231003-novavax"
            answers = [
                {
                    "questionId": q["questionId"],
                    "status": (
                        "OPEN" if self.unresolved_everything else (
                            "RESOLVED_BY_VISIBLE_SOURCE"
                            if any(s["sourceId"] == "fda-20230911-mrna"
                                   for s in visible) and
                            q["questionId"] == "fda-2023-xbb15-authorized-formula"
                            else "PARTIALLY_RESOLVED"
                        )
                    ),
                    "sourceId": (
                        "fda-20230911-mrna"
                        if any(s["sourceId"] == "fda-20230911-mrna"
                               for s in visible)
                        else q["originalVisibleSourceId"]
                    ),
                    "claimId": (
                        "mrna-authorization"
                        if any(s["sourceId"] == "fda-20230911-mrna"
                               for s in visible)
                        else q["originalVisibleClaimId"]
                    ),
                    "explanation": (
                        "The source statements now establish a separate "
                        "product authorization but do not prove financial effects."
                    ),
                    "stillUnknown": (
                        "Original manufacturer filing receipts and internal "
                        "regulatory review notes remain unavailable."
                    ),
                }
                for q in envelope["visibleResearchQuestions"]
            ]
            if self.omit_question:
                answers = []
            return json.dumps({
                "changeItems": changes, "questionAssessments": answers,
            })
        packet = envelope["visibleHistoricalEvidence"]
        draft = envelope["validatedModelDraft"]
        assert len(draft["modelAuthoredChangeItems"]) >= 2
        return json.dumps({
            "factInterpretations": [
                {
                    "sourceId": source["sourceId"],
                    "claimId": claim["claimId"],
                    "interpretation": (
                        "The official source described a distinct regulatory "
                        "action without proving realized manufacturer costs."
                    ),
                    "limitation": (
                        "This agency announcement is not a verified archived "
                        "publication-time snapshot or cost study."
                    ),
                }
                for source in packet["eligibleSources"]
                for claim in source["claims"]
            ],
            "edgeExplanations": [
                {
                    "edgeId": edge["edgeId"],
                    "fromSourceId": edge["fromSourceId"],
                    "toSourceId": edge["toSourceId"],
                    "interpretation": (
                        "A later official action followed an earlier regulator "
                        "announcement without proving economic causal effects."
                    ),
                    "missingEvidence": (
                        "Independent internal FDA decision records would be "
                        "needed to prove the complete reasoning mechanism."
                    ),
                }
                for edge in packet["visibleCuratedEdges"]
            ],
            "whatChanged": (
                "The regulator subsequently authorized distinct updated "
                "products after the earlier formula recommendation."
            ),
            "whatRemainsUnknown": (
                "Underlying manufacturer filing receipts and regulatory "
                "review meeting minutes are not present in the sources."
            ),
            "changeItems": [],
            "unresolvedQuestions": [],
        })


def test_staged_generation_covers_all_new_sources_and_keeps_provenance() -> None:
    doc, annotations = fda(), tracks()
    provider = StageProvider()
    run = chain.run_sequence(
        doc, [date(2023, 6, 17), date(2023, 10, 4)],
        provider, question_tracks=annotations, reasoning_mode="staged",
    )
    assert run["modelCalls"] == 4
    assert provider.calls == 4
    assert run["reasoningMode"] == "staged"
    early, late = run["replaySteps"]
    assert len(early["assessment"]["changeItems"]) == 2
    assert len(late["assessment"]["changeItems"]) == 2
    assert late["assessment"]["stageTwoChangeItems"] == []
    assert late["assessment"]["stageOneModelDraft"][
        "provenance"
    ] == "MODEL_GENERATED_AND_STRUCTURALLY_VALIDATED_NOT_SEMANTIC_PROOF"
    assert late["assessment"]["explanationQualityAudit"][
        "missingRecentSourceIds"
    ] == []
    assert len(late["assessment"]["modelQuestionAssessments"]) == 2
    assert run["historicalPITVerified"] is False
    assert run["generalizationVerified"] is False


def test_first_model_stage_cannot_see_future_sources_or_curator_answers() -> None:
    doc, annotations = fda(), tracks()
    provider = StageProvider()
    chain.run_sequence(
        doc, [date(2023, 6, 17), date(2023, 10, 4)],
        provider, question_tracks=annotations, reasoning_mode="staged",
    )
    early_input = provider.inputs[0]
    assert [x["sourceId"] for x in early_input[
        "visibleHistoricalEvidence"]["eligibleSources"]
    ] == ["fda-20230418-bivalent", "fda-20230616-formula"]
    assert "tracks" not in json.dumps(early_input)
    assert "RESOLVED_BY_LATER_SOURCE" not in json.dumps(early_input)
    assert "fda-20230911-mrna" not in json.dumps(early_input)
    assert "fda-20231003-novavax" not in json.dumps(early_input)
    assert len(early_input["visibleResearchQuestions"]) == 2
    assert len(provider.inputs[1]["validatedModelDraft"][
        "modelAuthoredQuestionAssessments"
    ]) == 2


@pytest.mark.parametrize("options,expected", [
    ({"skip_source": True}, "STAGED_DRAFT_CHANGE_EVIDENCE_INCOMPLETE"),
    ({"invent_future": True}, "STAGED_DRAFT_CHANGE_EVIDENCE_INCOMPLETE"),
    ({"omit_question": True}, "STAGED_DRAFT_QUESTION_REFERENCE_INVALID"),
])
def test_invalid_stage_one_is_rejected_before_narrative_call(
    options: dict, expected: str,
) -> None:
    provider = StageProvider(**options)
    with pytest.raises(ValueError, match=expected):
        chain.run_replay(
            fda(), date(2023, 6, 17), provider,
            question_tracks=tracks(), reasoning_mode="staged",
        )
    assert provider.calls == 1


def test_original_single_mode_is_unchanged() -> None:
    from test_historical_temporal_question_lifecycle import MinimalProvider

    baseline = chain.run_sequence(
        fda(), [date(2023, 6, 17), date(2023, 10, 4)],
        MinimalProvider(), question_tracks=tracks(),
    )
    assert baseline["modelCalls"] == 2
    assert baseline["reasoningMode"] == "single"
    assert baseline["replaySteps"][0]["assessment"]["changeItems"] == []


def test_visible_questions_only_uses_available_first_claims() -> None:
    doc, annotations = fda(), tracks()
    first = chain.prepare_replay(doc, date(2023, 4, 19))
    assert [x["questionId"] for x in visible_questions(annotations, first)] == [
        "fda-2023-xbb15-authorized-formula"
    ]
    old = copy.deepcopy(annotations)
    assert old == annotations
    assert EvidenceFirstDraft.model_json_schema()["properties"][
        "changeItems"
    ]["minItems"] == 1
