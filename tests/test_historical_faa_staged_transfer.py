"""FAA independent staged transfer and same-model FDA comparison: no Ollama."""
from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path

import pytest
from test_historical_cross_event_validation import fda
from test_historical_staged_reasoning import StageProvider
from test_historical_temporal_question_lifecycle import tracks as fda_tracks
from test_historical_official_event_chain import load as faa

from tris_market_event_intelligence import historical_official_event_chain_cli as chain
from tris_market_event_intelligence.historical_cross_event_staged_transfer_cli import (
    compare_independent_events,
    review_study,
)
from tris_market_event_intelligence.historical_staged_reasoning import (
    visible_questions,
)
from tris_market_event_intelligence.historical_temporal_question_lifecycle import (
    validate_question_tracks,
)

FAA_TRACKS = Path("research/faa-2024-question-lifecycle.json")


def faa_tracks() -> dict:
    return json.loads(FAA_TRACKS.read_text(encoding="utf-8"))


class FAAProvider:
    model_id = "mock-staged"

    def __init__(self) -> None:
        self.inputs: list[dict] = []
        self.calls = 0

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        self.calls += 1
        inp = json.loads(user)
        self.inputs.append(inp)
        packet = inp["visibleHistoricalEvidence"]
        if "questionAssessments" in schema["properties"]:
            fresh = [
                source for source in packet["eligibleSources"]
                if source["sourceId"] not in packet["previouslySeenSourceIds"]
            ]
            is_late = bool(packet["previouslySeenSourceIds"])
            decisions = []
            for question in inp["visibleResearchQuestions"]:
                key = question["questionId"]
                if is_late and "inspection-process" in key:
                    source_id, claim_id = (
                        "faa-20240124-production", "return-to-service",
                    )
                    status = "RESOLVED_BY_VISIBLE_SOURCE"
                elif is_late:
                    source_id, claim_id = (
                        "faa-20240124-production", "no-production-expansion",
                    )
                    status = "PARTIALLY_RESOLVED"
                else:
                    source_id = question["originalVisibleSourceId"]
                    claim_id = question["originalVisibleClaimId"]
                    status = "OPEN"
                decisions.append({
                    "questionId": key,
                    "status": status,
                    "sourceId": source_id,
                    "claimId": claim_id,
                    "explanation": (
                        "The relevant FAA source describes a concrete "
                        "regulatory oversight or inspection approval status."
                    ),
                    "stillUnknown": (
                        "Aircraft-specific inspection records, final audit "
                        "findings and manufacturing compliance remain unverified."
                    ),
                })
            return json.dumps({
                "changeItems": [{
                    "sourceId": source["sourceId"],
                    "claimId": source["claims"][0]["claimId"],
                    "changedAction": (
                        "The FAA announced a regulatory oversight or aircraft "
                        "inspection process action relevant to Boeing safety."
                    ),
                    "scope": (
                        "The Boeing aircraft model and inspection or production "
                        "oversight process identified in the official release."
                    ),
                } for source in fresh],
                "questionAssessments": decisions,
            })
        return json.dumps({
            "factInterpretations": [{
                "sourceId": source["sourceId"],
                "claimId": claim["claimId"],
                "interpretation": (
                    "This dated FAA announcement described an actual "
                    "aviation safety process or production oversight action."
                ),
                "limitation": (
                    "The record does not establish that each affected "
                    "airplane completed mandatory inspections."
                ),
            } for source in packet["eligibleSources"]
               for claim in source["claims"]],
            "edgeExplanations": [{
                "edgeId": e["edgeId"],
                "fromSourceId": e["fromSourceId"],
                "toSourceId": e["toSourceId"],
                "interpretation": (
                    "The later announcement followed the earlier FAA "
                    "statement, but causation cannot be proven from dates."
                ),
                "missingEvidence": (
                    "Independent quality control and inspection completion "
                    "reports would be necessary to prove the mechanism."
                ),
            } for e in packet["visibleCuratedEdges"]],
            "whatChanged": (
                "The FAA moved from proposed audit and inspection requirements "
                "toward final process approval and a production expansion hold."
            ),
            "whatRemainsUnknown": (
                "Specific inspection completion reports and finalized "
                "manufacturing audit findings remain unobserved in the record."
            ),
        })


def faa_run() -> tuple[dict, FAAProvider]:
    provider = FAAProvider()
    run = chain.run_sequence(
        faa(), [date(2024, 1, 13), date(2024, 1, 25)], provider,
        question_tracks=faa_tracks(), reasoning_mode="staged",
    )
    return run, provider


def test_faa_tracks_and_strict_time_scoped_model_input() -> None:
    original = faa()
    annotated = faa_tracks()
    chain.validate_chain(original)
    validate_question_tracks(original, annotated)
    assert len(annotated["tracks"]) == 2
    early = chain.prepare_replay(original, date(2024, 1, 13))
    late = chain.prepare_replay(original, date(2024, 1, 25))
    assert {x["sourceId"] for x in early["eligibleSources"]} == {
        "faa-20240112-oversight", "faa-20240112-inspection-process",
    }
    assert {x["sourceId"] for x in late["eligibleSources"]} == {
        "faa-20240112-oversight", "faa-20240112-inspection-process",
        "faa-20240121-900er-inspection", "faa-20240124-production",
    }
    assert all(
        x["sourceId"] != "faa-20240106-grounding"
        for x in early["eligibleSources"] + late["eligibleSources"]
    )
    run, provider = faa_run()
    assert run["modelCalls"] == 4
    assert provider.calls == 4
    first_stage = provider.inputs[0]
    assert first_stage["requiredNewSourceIds"] == [
        "faa-20240112-inspection-process", "faa-20240112-oversight",
    ]
    assert "faa-20240124-production" not in json.dumps(first_stage)
    assert "faa-20240121-900er-inspection" not in json.dumps(first_stage)
    assert "RESOLVED_BY_LATER_SOURCE" not in json.dumps(first_stage)
    assert len(visible_questions(annotated, early)) == 2
    assert run["historicalPITVerified"] is False


def test_faa_question_progress_and_distinct_aircraft_scope() -> None:
    run, provider = faa_run()
    early, late = run["replaySteps"]
    assert len(early["assessment"]["modelQuestionAssessments"]) == 2
    assert {x["status"] for x in early["assessment"][
        "modelQuestionAssessments"
    ]} == {"OPEN"}
    assert {x["status"] for x in late["assessment"][
        "modelQuestionAssessments"
    ]} == {"RESOLVED_BY_VISIBLE_SOURCE", "PARTIALLY_RESOLVED"}
    assert late["assessment"]["modelQuestionStatusReview"][
        "modelVsCuratorDisagreements"
    ] == []
    assert "faa-20240121-900er-inspection" in {
        x["sourceId"] for x in late["assessment"]["changeItems"]
    }
    assert "faa-20240124-production" in {
        x["sourceId"] for x in late["assessment"]["changeItems"]
    }
    assert "faa-20240106-grounding" not in json.dumps(
        provider.inputs[0]
    )


def test_independent_faa_vs_saved_fda_mock_transfer_has_zero_new_model_calls() -> None:
    reference = chain.run_sequence(
        fda(), [date(2023, 6, 17), date(2023, 10, 4)],
        StageProvider(), question_tracks=fda_tracks(), reasoning_mode="staged",
    )
    challenge, provider = faa_run()
    before = copy.deepcopy((reference, challenge))
    count = provider.calls
    report = compare_independent_events(
        fda(), reference, fda_tracks(), faa(), challenge, faa_tracks(),
    )
    assert provider.calls == count
    assert (reference, challenge) == before
    assert report["auditModelCalls"] == 0
    assert report["differentEventIdsVerified"] is True
    assert report["sameModelIdVerified"] is True
    assert report["referenceEvent"]["eventId"] != report[
        "independentChallengeEvent"
    ]["eventId"]
    assert report["independentChallengeEvent"][
        "newSourceCoverageComplete"
    ] is True
    assert report["independentChallengeEvent"][
        "noMechanicallyRejectedNarratives"
    ] is True
    assert report["generalizationVerified"] is False
    assert report["historicalPITVerified"] is False
    assert report["semanticAccuracyVerified"] is False


def test_transfer_rejects_same_event_forged_source_and_different_model() -> None:
    reference = chain.run_sequence(
        fda(), [date(2023, 6, 17), date(2023, 10, 4)],
        StageProvider(), question_tracks=fda_tracks(), reasoning_mode="staged",
    )
    challenge, _ = faa_run()
    with pytest.raises(ValueError, match="TRANSFER_EVENTS_MUST_BE_INDEPENDENT"):
        compare_independent_events(
            faa(), challenge, faa_tracks(), faa(), challenge, faa_tracks(),
        )
    forged = copy.deepcopy(challenge)
    forged["replaySteps"][0]["eligibleSources"].append(
        forged["replaySteps"][1]["eligibleSources"][-1]
    )
    with pytest.raises(ValueError, match="EVENT_CHAIN_SAVED_SOURCE_SCOPE_INVALID"):
        review_study(faa(), forged, faa_tracks())
    changed_model = copy.deepcopy(challenge)
    changed_model["replaySteps"][0]["modelId"] = "other"
    with pytest.raises(ValueError, match="TRANSFER_MODEL_ID_MISMATCH"):
        compare_independent_events(
            fda(), reference, fda_tracks(),
            faa(), changed_model, faa_tracks(),
        )
