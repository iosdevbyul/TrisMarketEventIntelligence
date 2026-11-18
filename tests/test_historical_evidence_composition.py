"""Source-scoped composition and independently grounded question axes."""
from __future__ import annotations

import json
from datetime import date

import pytest
from test_historical_cross_event_validation import fda
from test_historical_staged_reasoning import StageProvider
from test_historical_temporal_question_lifecycle import tracks

from tris_market_event_intelligence import historical_official_event_chain_cli as chain
from tris_market_event_intelligence.historical_evidence_composition import (
    ComposedEvidenceError,
)


class ComposedProvider(StageProvider):
    def __init__(self, *, wrong_scope: bool = False) -> None:
        super().__init__()
        self.wrong_scope = wrong_scope
        self.composed_inputs: list[dict] = []

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        properties = schema["properties"]
        if "items" in properties:
            self.calls += 1
            envelope = json.loads(user)
            self.inputs.append(envelope)
            self.composed_inputs.append(envelope)
            source = envelope["source"]
            return json.dumps({
                "items": [{
                    "sourceId": (
                        "future-invalid-source" if self.wrong_scope
                        else source["sourceId"]
                    ),
                    "claimId": claim["claimId"],
                    "explanation": (
                        "The official regulator documented a specific decision "
                        "for this source's named product and agency process."
                    ),
                    "limitation": (
                        "The document does not establish subsequent completed "
                        "implementation or underlying internal review records."
                    ),
                } for claim in source["claims"]],
            })
        if "assessments" in properties:
            self.calls += 1
            envelope = json.loads(user)
            self.inputs.append(envelope)
            questions = envelope["visibleQuestions"]
            return json.dumps({
                "assessments": [{
                    "questionId": q["questionId"],
                    "regulatoryActionProgress": "ANNOUNCED",
                    "requestedEvidenceStatus": "NOT_OBSERVED",
                    "sourceId": q["originalVisibleSourceId"],
                    "claimId": q["originalVisibleClaimId"],
                    "supportingEvidence": [{
                        "sourceId": q["originalVisibleSourceId"],
                        "claimId": q["originalVisibleClaimId"],
                    }],
                    "explanation": (
                        "An official statement describes the action but "
                        "the directly requested primary documents are absent."
                    ),
                    "remainingEvidence": (
                        "The full official underlying records and evidence "
                        "of completed implementation remain unavailable."
                    ),
                } for q in questions],
            })
        return super().generate_json(
            system=system, user=user, schema=schema,
        )


def test_composed_mode_has_source_provenance_and_independent_question_axes() -> None:
    provider = ComposedProvider()
    run = chain.run_sequence(
        fda(), [date(2023, 6, 17), date(2023, 10, 4)],
        provider, question_tracks=tracks(),
        reasoning_mode="staged-composed",
    )
    assert run["modelCalls"] == provider.calls
    assert run["modelCalls"] == 8
    for step in run["replaySteps"]:
        assessment = step["assessment"]
        composition = assessment["sourceScopedComposition"]
        assert len(composition["sourceExplanations"]) == 4
        assert composition["sourceCoverage"]["requiredSourceIds"] == (
            composition["sourceCoverage"]["explainedSourceIds"]
        )
        assert "VERBATIM_SOURCE_MODEL_SENTENCES" == composition["narrativeAssembly"]
        assert composition["sourceCoverage"]["requiredClaimRefs"] == (
            composition["sourceCoverage"]["explainedClaimRefs"]
        )
        axes = assessment["independentQuestionAxes"]["questionAssessments"]
        assert axes
        assert all(x["supportingEvidence"] for x in axes)
        assert all(x["requestedEvidenceStatus"] == "NOT_OBSERVED" for x in axes)
        assert assessment["questionAxesDoNotOverwriteLegacyStatuses"] is True
        assert assessment["semanticAccuracyVerified"] is False
        assert assessment["generationTerminationAnalysis"][
            "semanticCompletionVerified"
        ] is False
    early_input = json.dumps(provider.composed_inputs[0])
    assert "fda-20230911-mrna" not in early_input
    assert "RESOLVED_BY_LATER_SOURCE" not in early_input
    assert run["historicalPITVerified"] is False


def test_invalid_source_output_is_never_composed_or_filled_in() -> None:
    provider = ComposedProvider(wrong_scope=True)
    with pytest.raises(ComposedEvidenceError, match="COMPOSED_SOURCE_CLAIM_COVERAGE_INVALID"):
        chain.run_replay(
            fda(), date(2023, 6, 17), provider,
            question_tracks=tracks(), reasoning_mode="staged-composed",
        )
    assert provider.calls == 2


def test_question_axes_new_evidence_input_is_cutoff_scoped() -> None:
    provider = ComposedProvider()
    chain.run_sequence(
        fda(), [date(2023, 6, 17), date(2023, 10, 4)],
        provider, question_tracks=tracks(),
        reasoning_mode="staged-composed",
    )
    axis_inputs = [
        value for value in provider.inputs
        if "newlyExplainedClaims" in value
    ]
    assert len(axis_inputs) == 2
    assert len(axis_inputs[0]["newlyExplainedClaims"]) == 4
    assert "fda-20230911-mrna" not in json.dumps(axis_inputs[0])
    assert len(axis_inputs[1]["newlyExplainedClaims"]) == 4
