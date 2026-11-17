"""Compact staged generation, source scoping and transport termination metadata."""
from __future__ import annotations

import copy
import json
from datetime import date

import httpx
import pytest
from pydantic import ValidationError
from test_historical_cross_event_validation import fda
from test_historical_official_event_chain import load as faa
from test_historical_staged_reasoning import StageProvider
from test_historical_temporal_question_lifecycle import tracks as fda_tracks

from tris_market_event_intelligence import historical_official_event_chain_cli as chain
from tris_market_event_intelligence.historical_compact_narrative import (
    CompactNarrative,
    prepare_compact_input,
)
from tris_market_event_intelligence.llm_runtime import (
    ConfiguredLLMProvider,
    LLMCapabilities,
    LLMExecutionError,
    LLMSettings,
)


class CompactProvider(StageProvider):
    def __init__(self) -> None:
        super().__init__()
        self.compact_inputs: list[dict] = []
        self._diagnostic: dict | None = None

    @property
    def last_generation_diagnostics(self) -> dict | None:
        return self._diagnostic

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        self._diagnostic = {
            "provider": "mock",
            "doneReason": "stop",
            "generatedTokenCount": 50,
        }
        if set(schema["properties"]) == {
            "whatChanged", "whatRemainsUnknown",
        }:
            self.calls += 1
            payload = json.loads(user)
            self.inputs.append(payload)
            self.compact_inputs.append(payload)
            return json.dumps({
                "whatChanged": (
                    "The official regulator described a distinct approved "
                    "procedure and a separate oversight change affecting "
                    "the explicitly named products or aircraft."
                ),
                "whatRemainsUnknown": (
                    "The later operational completion evidence and original "
                    "regulatory review files remain unobserved."
                ),
            })
        return super().generate_json(
            system=system, user=user, schema=schema,
        )


def test_compact_mode_has_only_two_model_calls_per_cutoff_without_fake_facts() -> None:
    provider = CompactProvider()
    saved = chain.run_sequence(
        fda(), [date(2023, 6, 17), date(2023, 10, 4)],
        provider, question_tracks=fda_tracks(),
        reasoning_mode="staged-compact",
    )
    assert provider.calls == 4
    assert saved["modelCalls"] == 4
    assert saved["reasoningMode"] == "staged-compact"
    for step in saved["replaySteps"]:
        assessment = step["assessment"]
        assert assessment["individualFactInterpretationsGenerated"] is False
        assert assessment["individualEdgeExplanationsGenerated"] is False
        assert assessment["acceptedFactInterpretations"] == []
        assert assessment["acceptedEdgeExplanations"] == []
        assert assessment["sourceClaimCoverage"] == 0
        assert assessment["changeItems"] == assessment[
            "stageOneModelDraft"
        ]["modelAuthoredChangeItems"]
        assert assessment["stageTwoChangeItems"] == []
        assert assessment["stageTwoOutputContract"] == (
            "compact_narrative_two_fields_v2"
        )
        assert assessment["generationDiagnostics"]["stageTwo"][
            "doneReason"
        ] == "stop"
        assert assessment["stagedModelOutputReview"][
            "stageTwoExplanationUsable"
        ] is True
        assert assessment["semanticAccuracyVerified"] is False
    early = provider.compact_inputs[0]
    assert "fda-20230911-mrna" not in json.dumps(early)
    assert "RESOLVED_BY_LATER_SOURCE" not in json.dumps(early)
    assert len(early["newlyEligibleOfficialSources"]) == 2
    late = provider.compact_inputs[1]
    assert {x["sourceId"] for x in late["newlyEligibleOfficialSources"]} == {
        "fda-20230911-mrna", "fda-20231003-novavax",
    }
    assert "visibleResearchQuestions" not in late
    assert "questionLifecycle" not in late
    assert saved["historicalPITVerified"] is False


def test_compact_input_excludes_hindsight_and_retrospective_faa_source() -> None:
    doc = faa()
    packet = chain.prepare_replay(doc, date(2024, 1, 13))
    packet["previouslySeenSourceIds"] = []
    draft = {
        "modelAuthoredChangeItems": [],
        "modelAuthoredQuestionAssessments": [],
    }
    original = copy.deepcopy(packet)
    content = prepare_compact_input(packet, draft)
    assert packet == original
    assert {s["sourceId"] for s in content["newlyEligibleOfficialSources"]} == {
        "faa-20240112-oversight", "faa-20240112-inspection-process",
    }
    assert "faa-20240106-grounding" not in json.dumps(content)
    assert "faa-20240124-production" not in json.dumps(content)
    assert content["pointInTimeVerified"] is False


def test_compact_schema_allows_a_complete_explanation_longer_than_old_limit() -> None:
    schema = CompactNarrative.model_json_schema()
    assert schema["properties"]["whatChanged"]["maxLength"] == 1200
    longer = (
        "The FAA separately described inspection procedures, MAX production "
        "expansion limitations, and inspection recommendations for a "
        "different 737-900ER aircraft model. "
    ) * 5
    assert 600 < len(longer) <= 1200
    result = CompactNarrative.model_validate({
        "whatChanged": longer,
        "whatRemainsUnknown": (
            "The completion of each aircraft inspection and final production "
            "quality audit findings are not available in these releases."
        ),
    })
    assert result.whatChanged == longer


def test_saved_real_faa_compact_v1_boundary_truncation_is_rejected() -> None:
    from tris_market_event_intelligence.historical_staged_reasoning import (
        review_staged_model_output,
    )

    clipped = (
        "On January 21, 2024, the FAA recommended a visual inspection "
        "of door plugs on the Boeing 737-900ER aircraft, noting that this "
        "procedure is not applicable to the newer MAX fleet. The FAA also "
        "stated that the 737-900ER shares the door-plug design that "
        "motivated safety recommendations after the January 5 incident. "
        "On January 24, 2024, the FAA stated it would not grant Boeing "
        "any production expansion of the MAX, including the 737-9 MAX, "
        "while increasing oversight. Additionally, the FAA approved "
        "inspection and maintenance procedures for grounded 737-9 MAX "
        "airplanes, making them eligible to return a"
    )
    assert len(clipped) == 600
    result = {
        "stageTwoOutputContract": "compact_narrative_two_fields_v1",
        "whatChanged": clipped,
        "whatRemainsUnknown": (
            "The final production audit results and aircraft-by-aircraft "
            "inspection completion remain unseen."
        ),
        "stageOneModelDraft": {
            "modelAuthoredChangeItems": [],
            "modelAuthoredQuestionAssessments": [],
        },
    }
    old = copy.deepcopy(result)
    review = review_staged_model_output(
        chain.prepare_replay(faa(), date(2024, 1, 25)), result,
    )
    assert result == old
    assert review["schemaBoundaryCharacterLimit"] == 600
    assert review["narrativeCharacterCount"] == 600
    assert "STAGED_NARRATIVE_SCHEMA_BOUNDARY_TRUNCATION" in review["issues"]
    assert review["stageTwoExplanationUsable"] is False
    assert review["modelTextRewritten"] is False


def test_complete_old_boundary_and_new_short_prose_are_not_falsely_rejected() -> None:
    from tris_market_event_intelligence.historical_staged_reasoning import (
        review_staged_model_output,
    )

    text = "The regulator approved the 737-9 MAX inspection procedures."
    a = {
        "stageTwoOutputContract": "compact_narrative_two_fields_v2",
        "whatChanged": text,
        "whatRemainsUnknown": "Aircraft-by-aircraft completion is unknown.",
        "stageOneModelDraft": {
            "modelAuthoredChangeItems": [],
            "modelAuthoredQuestionAssessments": [],
        },
    }
    review = review_staged_model_output(
        chain.prepare_replay(faa(), date(2024, 1, 25)), a,
    )
    assert review["schemaBoundaryCharacterLimit"] == 1200
    assert review["issues"] == []


def test_compact_narrative_contract_rejects_extra_fake_ledger_fields() -> None:
    with pytest.raises(ValidationError):
        CompactNarrative.model_validate({
            "whatChanged": "The regulator approved a documented process.",
            "whatRemainsUnknown": (
                "The completion and implementation reports remain unseen."
            ),
            "factInterpretations": [{"sourceId": "invented"}],
        })


def _settings() -> LLMSettings:
    return LLMSettings(
        kind="ollama", model_id="gemma4:e2b",
        endpoint="http://127.0.0.1:11434",
        capabilities=LLMCapabilities(
            context_window=8192, max_output_tokens=4096,
        ),
    )


def test_ollama_transport_records_finish_reason_without_exposing_source_text() -> None:
    sent = []

    def respond(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={
            "done_reason": "stop",
            "prompt_eval_count": 1200,
            "eval_count": 410,
            "message": {"content": '{"whatChanged":"A sufficiently long example action string","whatRemainsUnknown":"Records remain unobserved after this action"}'},
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        provider = ConfiguredLLMProvider(_settings(), client=client)
        output = provider.generate_json(
            system="Only sourced output", user="date scoped public claims",
            schema=CompactNarrative.model_json_schema(),
        )
    assert json.loads(output)["whatChanged"].startswith("A sufficiently")
    diag = provider.last_generation_diagnostics
    assert diag["doneReason"] == "stop"
    assert diag["promptTokenCount"] == 1200
    assert diag["generatedTokenCount"] == 410
    assert diag["configuredOutputTokenLimit"] == 4096
    assert "date scoped" not in json.dumps(diag)
    assert sent[0]["options"]["num_predict"] == 4096


def test_ollama_length_termination_rejected_and_diagnosed() -> None:
    def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "done_reason": "length",
            "prompt_eval_count": 7900,
            "eval_count": 4096,
            "message": {"content": '{"whatChanged":"not complete"}'},
        })

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        provider = ConfiguredLLMProvider(_settings(), client=client)
        with pytest.raises(
            LLMExecutionError, match="MODEL_GENERATION_TRUNCATED",
        ):
            provider.generate_json(
                system="Strict", user="private source",
                schema=CompactNarrative.model_json_schema(),
            )
    diag = provider.last_generation_diagnostics
    assert diag["doneReason"] == "length"
    assert diag["generatedOutputLimitReached"] is True
    assert diag["generatedTokenCount"] == 4096


def test_openai_compatible_length_termination_does_not_return_partial_json() -> None:
    def respond(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{
                "finish_reason": "length",
                "message": {"content": "{}"},
            }],
        })
    config = LLMSettings(
        kind="openai-compatible", model_id="test-model",
        endpoint="http://127.0.0.1:9012/v1",
        api_key="fake",
    )
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        provider = ConfiguredLLMProvider(config, client=client)
        with pytest.raises(
            LLMExecutionError, match="MODEL_GENERATION_TRUNCATED",
        ):
            provider.generate_json(
                system="Strict", user="private source",
                schema=CompactNarrative.model_json_schema(),
            )
    assert provider.last_generation_diagnostics["finishReason"] == "length"
