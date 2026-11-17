"""Actual Gemma missing-source failure is diagnosable without weak acceptance."""
from __future__ import annotations

import copy
import json
import sys
from datetime import date
from pathlib import Path

import pytest
from test_historical_cross_event_validation import fda
from test_historical_staged_reasoning import StageProvider
from test_historical_temporal_question_lifecycle import tracks

from tris_market_event_intelligence import historical_official_event_chain_cli as chain
from tris_market_event_intelligence.historical_staged_reasoning import (
    StagedDraftEvidenceError,
    produce_draft,
)
from tris_market_event_intelligence.llm_runtime import LLMSettings


class MissingLateChangeProvider(StageProvider):
    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        data = json.loads(user)
        if "requiredNewSourceIds" in data:
            self.skip_source = bool(data["visibleHistoricalEvidence"][
                "previouslySeenSourceIds"
            ])
        return super().generate_json(system=system, user=user, schema=schema)


class DuplicateChangeProvider(StageProvider):
    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        raw = super().generate_json(system=system, user=user, schema=schema)
        data = json.loads(user)
        if "requiredNewSourceIds" in data:
            answer = json.loads(raw)
            answer["changeItems"][1] = answer["changeItems"][0]
            return json.dumps(answer)
        return raw


def test_first_stage_schema_requires_cutoff_source_slots_and_question_count() -> None:
    packet = chain.prepare_replay(fda(), date(2023, 6, 17))
    packet["previouslySeenSourceIds"] = []
    provider = StageProvider()
    draft = produce_draft(provider, packet, tracks())
    sent = provider.inputs[0]
    assert sent["requiredNewSourceIds"] == [
        "fda-20230418-bivalent", "fda-20230616-formula",
    ]
    assert provider.calls == 1
    assert len(draft["modelAuthoredChangeItems"]) == 2
    assert "fda-20230911-mrna" not in json.dumps(sent)
    assert "RESOLVED_BY_LATER_SOURCE" not in json.dumps(sent)


def test_diagnostic_reports_only_missing_source_and_submitted_ids() -> None:
    packet = chain.prepare_replay(fda(), date(2023, 6, 17))
    packet["previouslySeenSourceIds"] = []
    provider = StageProvider(skip_source=True)
    with pytest.raises(
        StagedDraftEvidenceError, match="STAGED_DRAFT_CHANGE_EVIDENCE_INCOMPLETE",
    ) as error:
        produce_draft(provider, packet, tracks())
    diag = error.value.diagnostic
    assert diag["reasonCodes"] == ["MISSING_NEW_SOURCE_CHANGE"]
    assert diag["missingNewSourceIds"] == ["fda-20230616-formula"]
    assert diag["requiredNewSourceIds"] == [
        "fda-20230418-bivalent", "fda-20230616-formula",
    ]
    assert diag["modelAuthoredTextNotPersisted"] is True
    assert error.value.completed_replay_steps == []
    assert provider.calls == 1


def test_duplicate_and_future_refs_have_specific_failure_codes() -> None:
    packet = chain.prepare_replay(fda(), date(2023, 6, 17))
    packet["previouslySeenSourceIds"] = []
    provider = DuplicateChangeProvider()
    with pytest.raises(StagedDraftEvidenceError) as error:
        produce_draft(provider, packet, tracks())
    assert error.value.diagnostic["reasonCodes"] == [
        "MISSING_NEW_SOURCE_CHANGE", "DUPLICATE_SOURCE_CLAIM",
    ]
    assert error.value.diagnostic["duplicateSourceClaimIds"] == [{
        "sourceId": "fda-20230418-bivalent", "claimId": "schedule-simplified",
    }]


def test_late_failure_retains_earlier_cutoff_and_attempt_count() -> None:
    provider = MissingLateChangeProvider()
    with pytest.raises(StagedDraftEvidenceError) as error:
        chain.run_sequence(
            fda(), [date(2023, 6, 17), date(2023, 10, 4)], provider,
            question_tracks=tracks(), reasoning_mode="staged",
        )
    exc = error.value
    assert provider.calls == 3
    assert exc.diagnostic["failedAsOfDateExclusive"] == "2023-10-04"
    assert exc.diagnostic["modelCallsAttempted"] == 3
    assert exc.diagnostic["completedCutoffCount"] == 1
    assert len(exc.completed_replay_steps) == 1
    assert exc.completed_replay_steps[0]["asOfDateExclusive"] == "2023-06-17"
    assert exc.diagnostic["missingNewSourceIds"] == ["fda-20231003-novavax"]


def test_cli_persists_explicit_failed_status_and_no_fake_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider = MissingLateChangeProvider()
    output = tmp_path / "failed-run.json"
    monkeypatch.setattr(
        chain.LLMSettings, "from_environment",
        lambda **_: LLMSettings(
            kind="ollama", model_id="gemma4:e2b",
            endpoint="http://127.0.0.1:11434",
        ),
    )
    monkeypatch.setattr(
        chain, "build_runtime", lambda *_args, **_kwargs: provider,
    )
    monkeypatch.setattr(sys, "argv", [
        "historical-event-chain",
        "--chain", "research/fda-covid-vaccine-2023-event-chain.json",
        "--question-tracks", "research/fda-2023-question-lifecycle.json",
        "--sequence", "2023-06-17", "2023-10-04",
        "--reasoning-mode", "staged", "--run", "--model", "gemma4:e2b",
        "--output", str(output),
    ])
    chain.main()
    printed = json.loads(capsys.readouterr().out)
    assert printed["analysisSucceeded"] is False
    assert printed["status"] == "STAGED_DRAFT_EVIDENCE_REJECTED"
    assert printed["modelCallsAttempted"] == 3
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["allStepsCompleted"] is False
    assert saved["modelCallsAttempted"] == 3
    assert len(saved["completedReplaySteps"]) == 1
    assert saved["failure"]["missingNewSourceIds"] == [
        "fda-20231003-novavax",
    ]
    assert "modelAuthoredChangeItems" not in json.dumps(saved["failure"])
    original = copy.deepcopy(saved)
    with pytest.raises(ValueError, match="Expected new private output"):
        chain.main()
    assert json.loads(output.read_text(encoding="utf-8")) == original


def test_single_cutoff_failure_also_creates_diagnostic_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    provider = StageProvider(skip_source=True)
    output = tmp_path / "single-failure.json"
    monkeypatch.setattr(
        chain.LLMSettings, "from_environment",
        lambda **_: LLMSettings(
            kind="ollama", model_id="gemma4:e2b",
            endpoint="http://127.0.0.1:11434",
        ),
    )
    monkeypatch.setattr(
        chain, "build_runtime", lambda *_args, **_kwargs: provider,
    )
    monkeypatch.setattr(sys, "argv", [
        "historical-event-chain",
        "--chain", "research/fda-covid-vaccine-2023-event-chain.json",
        "--question-tracks", "research/fda-2023-question-lifecycle.json",
        "--as-of", "2023-06-17", "--reasoning-mode", "staged",
        "--run", "--model", "gemma4:e2b", "--output", str(output),
    ])
    chain.main()
    result = json.loads(capsys.readouterr().out)
    assert result["analysisSucceeded"] is False
    assert result["modelCallsAttempted"] == 1
    assert provider.calls == 1
    assert json.loads(output.read_text(encoding="utf-8"))[
        "status"
    ] == "STAGED_DRAFT_EVIDENCE_REJECTED"
