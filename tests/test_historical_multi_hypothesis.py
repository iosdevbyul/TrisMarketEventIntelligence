"""Human-guided research method is in-context, never an LLM weight update."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_historical_story_review import _fixture

from tris_market_event_intelligence.historical_multi_hypothesis import (
    PRIMARY_ID,
    build_research_context,
    compare_historical_hypotheses,
    read_reviewed_lessons,
    research_from_scout,
)
from tris_market_event_intelligence.historical_story_review import (
    prepare_historical_story_review,
)


class FakeProvider:
    model_id = "gemma4:e2b"

    def __init__(self, *, unsupported: bool = False, proof: bool = False) -> None:
        self.unsupported = unsupported
        self.proof = proof
        self.calls = 0
        self.seen = None
        self.system = ""

    def generate_json(
        self, *, system: str, user: str, schema: dict[str, object] | None,
    ) -> str:
        self.calls += 1
        self.seen = json.loads(user)
        self.system = system
        assert "GOOD REASONING EXAMPLE" in system
        assert schema is not None
        assert "stockReturn" not in user
        assert len(self.seen["urlHintRows"]) == 4
        assert {row[0] for row in self.seen["urlHintRows"]} == {
            "ref-001", "ref-002", "ref-003", "ref-004"
        }
        assert self.seen["macroVintageAvailable"] is False
        assert self.seen["priorOutcomeNotProvided"] is True
        evidence = ("ref-999" if self.unsupported else PRIMARY_ID)
        return json.dumps({
            "hypotheses": [
                {
                    "category": "COMPANY_EARNINGS_OR_GUIDANCE",
                    "reference_ids": [evidence, "ref-001"],
                    "mechanism": "Forward demand expectations could influence valuation.",
                    "alternative_explanations": "Semiconductor cycle conditions may also matter.",
                    "source_limitations": "Source announcement trading clock remains unverified.",
                    "evidence_strength": "PRIMARY_DOCUMENT",
                },
                {
                    "category": "COMPETITOR_OR_SECTOR",
                    "reference_ids": ["ref-003"],
                    "mechanism": "Competing launches may affect product share expectations.",
                    "alternative_explanations": "Company guidance provides a competing explanation.",
                    "source_limitations": "Only URL-derived hints, not actual article text.",
                    "evidence_strength": "URL_HINT_ONLY",
                },
                {
                    "category": "UNASSESSED",
                    "reference_ids": [],
                    "mechanism": "Macroeconomic contribution cannot be established yet.",
                    "alternative_explanations": "Policy and broader market news might be confounders.",
                    "source_limitations": "No verified macro vintage or true release clock.",
                    "evidence_strength": "INSUFFICIENT",
                },
            ],
            "research_summary": "An issuer document merits separate review.",
            "missing_evidence": "Original publisher publication clocks and macro vintages.",
            "causal_proof": self.proof,
            "prospective_prediction": False,
        })


def test_all_urls_and_primary_are_shown_without_future_returns(tmp_path: Path) -> None:
    study = _fixture(tmp_path)
    story = prepare_historical_story_review(study)
    context = build_research_context(story=story)
    ids = {
        row["referenceId"]
        for rows in context["referencesByHeuristicCategory"].values()
        for row in rows
    }
    assert ids == {"ref-001", "ref-002", "ref-003", "ref-004"}
    assert context["allReferencesProvided"] == 4
    assert context["outcomeMaskedFromModel"] is True
    assert context["economicIndicatorsActuallyAvailable"] is False
    assert context["firstPartyDisclosureSeparateFromUrlHints"]["referenceId"] == PRIMARY_ID


def test_gemma_compares_three_distinct_research_candidates(tmp_path: Path) -> None:
    provider = FakeProvider()
    result = research_from_scout(_fixture(tmp_path), provider=provider)
    assert provider.calls == 1
    assert result["status"] == "RETROSPECTIVE_HYPOTHESES_UNVERIFIED"
    assert result["allCandidateReferenceCount"] == 4
    assert result["modelCalls"] == 1
    assert len(result["hypotheses"]) == 3
    assert result["hypotheses"][0]["reference_ids"][0] == PRIMARY_ID
    assert result["weightTrainingPerformed"] is False
    assert result["forecastImprovementMeasured"] is False
    assert result["independentGoldCreated"] is False


def test_model_cannot_invent_sources_or_certify_causation(tmp_path: Path) -> None:
    study = prepare_historical_story_review(_fixture(tmp_path))
    with pytest.raises(ValueError, match="Unsupported source"):
        compare_historical_hypotheses(story=study, provider=FakeProvider(unsupported=True))
    with pytest.raises(ValueError, match="cannot certify"):
        compare_historical_hypotheses(story=study, provider=FakeProvider(proof=True))


def test_human_lessons_required_and_separately_pinned(tmp_path: Path) -> None:
    path = tmp_path / "lessons.json"
    path.write_text(json.dumps({
        "schemaVersion": "human_reviewed_research_lessons_v1",
        "reviewedByHuman": True,
        "lessons": [
            {"lesson": "Compare direct issuer guidance against sector stories.",
             "evidence": "An official dated company earnings release exists."},
        ],
    }))
    lessons, digest = read_reviewed_lessons(path)
    assert digest and len(digest) == 64
    provider = FakeProvider()
    result = research_from_scout(
        _fixture(tmp_path), provider=provider, reviewed_lessons=path,
    )
    assert result["usedHumanReviewedLessons"] is True
    assert result["humanReviewedLessonsSha256"] == digest
    assert provider.seen["reviewedLessons"] == lessons
    assert result["weightTrainingPerformed"] is False
    tampered = json.loads(path.read_text())
    tampered["reviewedByHuman"] = False
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="human-reviewed"):
        read_reviewed_lessons(path)


def test_model_output_cannot_be_used_as_learning_memory(tmp_path: Path) -> None:
    path = tmp_path / "model_output.json"
    path.write_text(json.dumps({
        "schemaVersion": "historical_story_hypothesis_v1",
        "status": "MODEL_HYPOTHESIS_UNVERIFIED",
        "modelCalls": 1,
    }))
    with pytest.raises(ValueError, match="human-reviewed"):
        read_reviewed_lessons(path)


def test_compact_model_input_contains_44_sources_without_repeating_caveats(
    tmp_path: Path,
) -> None:
    from tris_market_event_intelligence.historical_multi_hypothesis import (
        _compact_model_context,
    )

    story = prepare_historical_story_review(_fixture(tmp_path))
    source = story["references"][0]
    while len(story["references"]) < 44:
        next_ref = dict(source)
        next_ref["referenceId"] = f"ref-{len(story['references']) + 1:03d}"
        next_ref["urlDerivedHint"] = "nvidia quarterly earnings revenue outlook ai"
        story["references"].append(next_ref)
    context = build_research_context(story=story)
    slim = _compact_model_context(context)
    assert len(slim["urlHintRows"]) == 44
    assert len({row[0] for row in slim["urlHintRows"]}) == 44
    assert "publisherFirstPublishedAt" not in json.dumps(slim)
    assert len(json.dumps(slim).encode()) < 12000
    assert slim["sourceLimitations"]


def test_mac_cli_configures_finite_generation_budget_for_local_ollama(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    import sys

    from tris_market_event_intelligence import historical_multi_hypothesis_cli as cli
    from tris_market_event_intelligence.llm_runtime import LLMSettings

    monkeypatch.setattr(
        cli.LLMSettings, "from_environment",
        classmethod(lambda cls, explicit=False: LLMSettings(
            kind="ollama", model_id="gemma4:e2b",
            endpoint="http://127.0.0.1:11434",
        )),
    )
    captured = {}
    def build_local(settings: LLMSettings, *, confirm_model: bool = False) -> FakeProvider:
        assert confirm_model is True
        captured["context"] = settings.capabilities.context_window
        captured["output"] = settings.capabilities.max_output_tokens
        captured["thinking"] = settings.capabilities.thinking
        return FakeProvider()
    monkeypatch.setattr(cli, "build_runtime", build_local)
    output = tmp_path / "historical-result.json"
    monkeypatch.setattr(sys, "argv", [
        "historical-multi", "--scout", str(_fixture(tmp_path)),
        "--run", "--model", "gemma4:e2b", "--output", str(output),
    ])
    cli.main()
    assert captured == {"context": 8192, "output": 3072, "thinking": False}
    data = json.loads(output.read_text())
    assert len(data["hypotheses"]) == 3
    assert data["weightTrainingPerformed"] is False
    assert data["allCandidateReferenceCount"] == 4
    assert '"outputTokensConfigured": 3072' in capsys.readouterr().out
