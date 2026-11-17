"""Real FDA stage-two gibberish detection and slim narrative contract."""
from __future__ import annotations

import copy
import json
from datetime import date

from test_historical_cross_event_validation import fda
from test_historical_staged_reasoning import StageProvider
from test_historical_temporal_question_lifecycle import tracks

from tris_market_event_intelligence import historical_official_event_chain_cli as chain
from tris_market_event_intelligence.historical_staged_reasoning import (
    review_staged_model_output,
)
from tris_market_event_intelligence.historical_temporal_question_lifecycle import (
    audit_saved_temporal,
)


class NarrativeProvider(StageProvider):
    def __init__(self, *, corrupt: bool = False, malformed_unknown: bool = False):
        super().__init__()
        self.corrupt = corrupt
        self.malformed_unknown = malformed_unknown
        self.phase_two_schemas: list[dict] = []

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        if "questionAssessments" in schema["properties"]:
            raw = json.loads(super().generate_json(
                system=system, user=user, schema=schema,
            ))
            if self.malformed_unknown:
                raw["questionAssessments"][0]["stillUnknown"] = (
                    'false, "sourceId": "fda-20230911-mrna", "claimId": "mrna-authorization"'
                )
            return json.dumps(raw)
        self.phase_two_schemas.append(schema)
        raw = json.loads(super().generate_json(
            system=system, user=user, schema=schema,
        ))
        if self.corrupt:
            raw["whatChanged"] = (
                "modelAuthoredChangeItems_" + "_".join(
                    str(i) for i in range(1, 101)
                )
            )
        return json.dumps(raw)


def test_actual_fda_corrupt_october_narrative_is_rejected_not_normalized() -> None:
    doc, annotations = fda(), tracks()
    provider = NarrativeProvider(corrupt=True, malformed_unknown=True)
    saved = chain.run_sequence(
        doc, [date(2023, 6, 17), date(2023, 10, 4)],
        provider, question_tracks=annotations, reasoning_mode="staged",
    )
    assert saved["modelCalls"] == 4
    assert saved["status"] == "HISTORICAL_SEQUENCE_CONTAINS_REJECTED_NARRATIVE"
    for step in saved["replaySteps"]:
        review = step["assessment"]["stagedModelOutputReview"]
        assert review["stageTwoExplanationUsable"] is False
        assert "STAGED_NARRATIVE_MECHANICALLY_CORRUPTED" in review["issues"]
        assert "STAGED_QUESTION_REMAINDER_IS_NOT_NATURAL_LANGUAGE" in review["issues"]
        assert step["assessment"]["whatChanged"].startswith(
            "modelAuthoredChangeItems_1_2_3"
        )
        assert step["overallStatus"] == "STAGED_NARRATIVE_GENERATION_REJECTED"
    assert saved["semanticAccuracyVerified"] is False
    assert all("changeItems" not in schema["properties"] and
               "unresolvedQuestions" not in schema["properties"]
               for schema in provider.phase_two_schemas)


def test_saved_legacy_staged_output_can_be_reviewed_without_any_model_calls() -> None:
    doc, annotations = fda(), tracks()
    provider = NarrativeProvider(corrupt=True, malformed_unknown=True)
    original = chain.run_sequence(
        doc, [date(2023, 6, 17), date(2023, 10, 4)],
        provider, question_tracks=annotations, reasoning_mode="staged",
    )
    untouched = copy.deepcopy(original)
    count = provider.calls
    result = audit_saved_temporal(doc, original, annotations)
    assert provider.calls == count
    assert original == untouched
    assert result["auditModelCalls"] == 0
    assert all(
        x["savedStagedModelOutputReview"]["stageTwoExplanationUsable"] is False
        for x in result["replayStepAudits"]
    )


def test_valid_source_grounded_prose_is_reviewable_without_claiming_truth() -> None:
    provider = NarrativeProvider()
    saved = chain.run_sequence(
        fda(), [date(2023, 6, 17), date(2023, 10, 4)],
        provider, question_tracks=tracks(), reasoning_mode="staged",
    )
    assert saved["status"] == "HISTORICAL_DATE_ONLY_SEQUENCE_REQUIRES_HUMAN_REVIEW"
    assert all(x["assessment"]["stagedModelOutputReview"][
        "stageTwoExplanationUsable"
    ] for x in saved["replaySteps"])
    assert all(not x["assessment"]["stagedModelOutputReview"][
        "semanticAccuracyVerified"
    ] for x in saved["replaySteps"])


def test_original_user_observed_question_field_json_fragment_is_flagged() -> None:
    packet = chain.prepare_replay(fda(), date(2023, 10, 4))
    assessment = {
        "stageOneModelDraft": {
            "modelAuthoredChangeItems": [
                {"sourceId": "fda-20230911-mrna"},
                {"sourceId": "fda-20231003-novavax"},
            ],
            "modelAuthoredQuestionAssessments": [
                {
                    "questionId": "fda-2023-xbb15-authorized-formula",
                    "stillUnknown": 'false, "sourceId": "fda-20230911-mrna"',
                },
            ],
        },
        "whatChanged": (
            "FDA authorized updated mRNA products from ModernaTX and Pfizer "
            "then the distinct Novavax vaccine for the 2023 season."
        ),
        "whatRemainsUnknown": (
            "The internal FDA review records and manufacturer submission "
            "receipts are not visible in these source announcements."
        ),
    }
    review = review_staged_model_output(packet, assessment)
    assert review["stageTwoExplanationUsable"] is True
    assert review["issues"] == ["STAGED_QUESTION_REMAINDER_IS_NOT_NATURAL_LANGUAGE"]
    assert review["semanticAccuracyVerified"] is False
