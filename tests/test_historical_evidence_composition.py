"""Source-scoped composition and independently grounded question axes."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

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
        if "explanation" in properties and "limitation" in properties:
            self.calls += 1
            envelope = json.loads(user)
            self.inputs.append(envelope)
            self.composed_inputs.append(envelope)
            if self.wrong_scope:
                return json.dumps({
                    "explanation": "The regulator described a source action without finishing the sentence",
                    "limitation": (
                        "The original official claim does not establish "
                        "the outcome of the following stage."
                    ),
                })
            return json.dumps({
                "explanation": (
                    "The official regulator documented the concrete action "
                    "described in this single official claim, with an "
                    "explicitly defined product and agency scope."
                ),
                "limitation": (
                    "The original official claim does not establish "
                    "the final outcome of later implementation."
                ),
            })
        if properties and all(
            key.startswith("claim_") for key in properties
        ):
            self.calls += 1
            envelope = json.loads(user)
            self.inputs.append(envelope)
            return json.dumps({
                key: (
                    "RELATED_CONTEXT" if index == 0 else "NOT_RELEVANT"
                )
                for index, key in enumerate(properties)
            })
        if "regulatoryActionProgress" in properties:
            self.calls += 1
            envelope = json.loads(user)
            self.inputs.append(envelope)
            return json.dumps({
                "regulatoryActionProgress": "ANNOUNCED",
                "requestedEvidenceStatus": "PARTIAL",
                "explanation": (
                    "The official source describes a regulatory step but "
                    "does not establish final action or completed records."
                ),
                "remainingEvidence": (
                    "The final supporting documents and complete "
                    "implementation outcomes remain unobserved."
                ),
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
    assert run["modelCalls"] == 18
    for step in run["replaySteps"]:
        assessment = step["assessment"]
        composition = assessment["sourceScopedComposition"]
        assert len(composition["sourceExplanations"]) == 4
        assert composition["sourceCoverage"]["requiredSourceIds"] == (
            composition["sourceCoverage"]["explainedSourceIds"]
        )
        assert "VERBATIM_CLAIM_MODEL_SENTENCES" == composition["narrativeAssembly"]
        assert composition["sourceCoverage"]["requiredClaimRefs"] == (
            composition["sourceCoverage"]["explainedClaimRefs"]
        )
        axes = assessment["independentQuestionAxes"]["questionAssessments"]
        assert axes
        assert all(x["supportingEvidence"] for x in axes)
        assert all(x["requestedEvidenceStatus"] == "PARTIAL" for x in axes)
        assert len(assessment["independentQuestionAxes"]["questionEvidenceScreens"]) == 2
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
    with pytest.raises(ComposedEvidenceError, match="COMPOSED_SOURCE_EXPLANATION_INCOMPLETE"):
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
    screen_inputs = [
        value for value in provider.inputs
        if "indexedVisibleOfficialClaims" in value
    ]
    answer_inputs = [
        value for value in provider.inputs
        if "screenedRelevantEvidence" in value
    ]
    assert len(screen_inputs) == len(answer_inputs) == 4
    assert all(len(value["indexedVisibleOfficialClaims"]) == 4
               for value in screen_inputs[:2])
    assert all(len(value["indexedVisibleOfficialClaims"]) == 8
               for value in screen_inputs[2:])
    assert "fda-20230911-mrna" not in json.dumps(screen_inputs[:2])


def test_per_claim_provenance_is_bound_to_visible_input_not_model_ids() -> None:
    provider = ComposedProvider()
    step = chain.run_replay(
        fda(), date(2023, 6, 17), provider,
        question_tracks=tracks(), reasoning_mode="staged-composed",
    )
    composition = step["assessment"]["sourceScopedComposition"]
    assert composition["claimProvenance"] == "ASSIGNED_FROM_INPUT_NOT_MODEL_GENERATED"
    assert step["modelCalls"] == 9
    assert len(provider.composed_inputs) == 4
    assert all("claim" in value for value in provider.composed_inputs)
    assert all(len(source["claimCalls"]) == 2 for source in
               composition["sourceGenerationDiagnostics"])
    assert all(
        len(group["claims"]) == 2
        for group in step["assessment"]["generationTerminationBySource"]
    )


class DecisionScreenProvider(ComposedProvider):
    """Synthetic selection fixtures for the source-review pipeline."""

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        properties = schema["properties"]
        if properties and all(k.startswith("claim_") for k in properties):
            self.calls += 1
            envelope = json.loads(user)
            self.inputs.append(envelope)
            question = envelope["question"]["questionId"]
            roles = {}
            for idx, source in enumerate(envelope["indexedVisibleOfficialClaims"]):
                ref = (source["sourceId"], source["claimId"])
                role = "NOT_RELEVANT"
                if "boeing-production-oversight" in question:
                    if ref == ("faa-20240112-oversight", "production-audit"):
                        role = "RELATED_CONTEXT"
                    if ref == ("faa-20240124-production", "no-production-expansion"):
                        role = "INTERMEDIATE_DECISION"
                elif "manufacturer-filings" in question:
                    if ref == ("fda-20230616-formula", "filing-still-needed"):
                        role = "RELATED_CONTEXT"
                    if ref == ("fda-20230911-mrna", "mrna-authorization"):
                        role = "INTERMEDIATE_DECISION"
                elif "inspection-process-approval" in question:
                    if ref == ("faa-20240112-inspection-process", "additional-data"):
                        role = "RELATED_CONTEXT"
                    if ref == ("faa-20240124-production", "return-to-service"):
                        role = "DIRECT_ANSWER"
                elif "authorized-formula" in question:
                    if ref == ("fda-20230616-formula", "xbb-recommendation"):
                        role = "RELATED_CONTEXT"
                    if ref == ("fda-20230911-mrna", "mrna-authorization"):
                        role = "DIRECT_ANSWER"
                roles[f"claim_{idx}"] = role
            return json.dumps(roles)
        if "regulatoryActionProgress" in properties:
            self.calls += 1
            envelope = json.loads(user)
            self.inputs.append(envelope)
            return json.dumps({
                "regulatoryActionProgress": "AUTHORIZED"
                if any(x["role"] == "DIRECT_ANSWER" for x
                       in envelope["screenedRelevantEvidence"]) else "ANNOUNCED",
                "requestedEvidenceStatus": "DIRECTLY_OBSERVED"
                if any(x["role"] == "DIRECT_ANSWER" for x
                       in envelope["screenedRelevantEvidence"]) else "PARTIAL",
                "explanation": (
                    "The regulator announced an intermediate decision or "
                    "authorization, but a separate operational completion "
                    "or original-review document may remain unavailable."
                ),
                "remainingEvidence": (
                    "Underlying operational completion and detailed agency "
                    "review records have not been independently verified."
                ),
            })
        return super().generate_json(system=system, user=user, schema=schema)


def test_later_official_decisions_are_reconciled_separately_from_final_records() -> None:
    from test_historical_official_event_chain import load as faa
    from test_historical_temporal_question_lifecycle import tracks as fda_tracks

    for chain_doc, question_tracks, cutoff, question, expected in (
        (fda(), fda_tracks(), date(2023, 10, 4),
         "fda-2023-manufacturer-filings-evidence",
         ("fda-20230911-mrna", "mrna-authorization")),
        (faa(), json.loads(Path("research/faa-2024-question-lifecycle.json").read_text(encoding="utf-8")), date(2024, 1, 25),
         "faa-2024-boeing-production-oversight-outcome",
         ("faa-20240124-production", "no-production-expansion")),
    ):
        provider = DecisionScreenProvider()
        step = chain.run_replay(
            chain_doc, cutoff, provider,
            question_tracks=question_tracks, reasoning_mode="staged-composed",
        )
        assessment = step["assessment"]["independentQuestionAxes"]
        item = next(x for x in assessment["questionAssessments"]
                    if x["questionId"] == question)
        assert (item["sourceId"], item["claimId"]) == expected
        assert item["requestedEvidenceStatus"] == "PARTIAL"
        assert expected in {
            (x["sourceId"], x["claimId"])
            for x in item["supportingEvidence"]
        }
        assert assessment["axesSemanticallyVerified"] is False


class InvalidQuestionAnswerProvider(ComposedProvider):
    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        if "regulatoryActionProgress" in schema["properties"]:
            self.calls += 1
            envelope = json.loads(user)
            self.inputs.append(envelope)
            return json.dumps({
                "regulatoryActionProgress": "ANNOUNCED",
                "requestedEvidenceStatus": "PARTIAL",
                "explanation": (
                    "The available source documents a regulator action "
                    "but does not establish the completed review process"
                ),
                "remainingEvidence": (
                    "Final underlying review records were not available "
                    "in the quoted regulatory source."
                ),
            })
        return super().generate_json(system=system, user=user, schema=schema)


def test_invalid_question_answer_identifies_question_and_field() -> None:
    provider = InvalidQuestionAnswerProvider()
    with pytest.raises(
        ComposedEvidenceError,
        match=r"COMPOSED_QUESTION_ANSWER_INVALID:.*:EXPLANATION_INCOMPLETE",
    ):
        chain.run_replay(
            fda(), date(2023, 6, 17), provider,
            question_tracks=tracks(), reasoning_mode="staged-composed",
        )


class OnceIncompleteQuestionProvider(ComposedProvider):
    def __init__(self) -> None:
        super().__init__()
        self.failed_question_once = False

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        if ("regulatoryActionProgress" in schema["properties"]
                and not self.failed_question_once):
            self.failed_question_once = True
            self.calls += 1
            self.inputs.append(json.loads(user))
            return json.dumps({
                "regulatoryActionProgress": "ANNOUNCED",
                "requestedEvidenceStatus": "PARTIAL",
                "explanation": (
                    "The source describes a specific regulator action "
                    "without final completion information"
                ),
                "remainingEvidence": (
                    "The final decision records were not present in "
                    "the dated official release."
                ),
            })
        return super().generate_json(system=system, user=user, schema=schema)


def test_one_invalid_question_answer_is_regenerated_and_counted() -> None:
    provider = OnceIncompleteQuestionProvider()
    step = chain.run_replay(
        fda(), date(2023, 6, 17), provider,
        question_tracks=tracks(), reasoning_mode="staged-composed",
    )
    assert provider.failed_question_once
    assert step["modelCalls"] == provider.calls == 10
    diagnostics = step["assessment"]["independentQuestionAxes"]["generationDiagnostics"]
    assert len(diagnostics["relevanceScreenCalls"]) == 2
    assert len(diagnostics["questionAnswerCalls"]) == 3
    assert step["assessment"]["semanticAccuracyVerified"] is False
