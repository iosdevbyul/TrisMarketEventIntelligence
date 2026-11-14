"""V8 market transmission requirements derived from a real Gemma 2 failure."""
import json

import httpx
import pytest

from tris_market_event_intelligence.impact import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    EventInput,
    Impact,
    ImpactAnalysisError,
    ImpactResult,
    OpenAICompatibleProvider,
    event_specific_instructions,
    validate_result,
)
from tris_market_event_intelligence.impact_mechanism import mechanism_quality_code


SOURCE = "□ 2026년 2/4분기 외감기업의 성장성, 수익성, 안정성이 모두 개선"


def event() -> EventInput:
    return EventInput(
        13, "EARNINGS", "UNVERIFIED", None, None, SOURCE,
        "official-release", "https://example.org/e", "now", "REPORTED",
    )


def hypothesis(rationale: str, sector: str = "BROAD_MARKET") -> ImpactResult:
    return ImpactResult(impacts=[Impact(
        market="GLOBAL", sector=sector, direction="UNCERTAIN",
        horizon="MEDIUM_TERM", rationale=rationale, evidence_excerpt=SOURCE,
        uncertainty="HIGH", drivers=[], entities=["외감기업"],
    )])


def test_prompt_version_and_anti_summary_contract() -> None:
    assert PROMPT_VERSION == "market-impact-v8"
    assert "transmission" in SYSTEM_PROMPT.lower()
    assert "NO_HYPOTHESIS" not in SYSTEM_PROMPT
    assert "impacts" in event_specific_instructions(event())


@pytest.mark.parametrize("rationale", [
    "Improved financial performance may indicate broader economic strength.",
    "기업들의 성장성과 수익성이 개선됐으므로 경제가 강해질 수 있습니다.",
    "The financial results appear to be better than expected.",
])
def test_broad_market_generic_macro_summary_rejected(rationale: str) -> None:
    with pytest.raises(ImpactAnalysisError) as exc:
        validate_result(event(), hypothesis(rationale))
    assert exc.value.code == "MARKET_MECHANISM_MISSING"


@pytest.mark.parametrize("rationale", [
    (
        "If the reported improvement persists, investor earnings expectations "
        "might improve equity valuations, but surprise and pricing are unknown."
    ),
    (
        "보고된 수익성 개선이 지속된다면 주식시장의 기업이익 기대에 영향을 "
        "줄 수 있지만 기존 반영 여부는 알 수 없습니다."
    ),
])
def test_explicit_conditional_broad_market_channel_passes_syntax_only(
    rationale: str,
) -> None:
    assert validate_result(event(), hypothesis(rationale)).impacts
    assert mechanism_quality_code("BROAD_MARKET", rationale) is None


def test_no_hypothesis_and_other_sector_not_forced_to_invent_mechanism() -> None:
    assert validate_result(event(), ImpactResult(impacts=[])).impacts == []
    assert mechanism_quality_code(
        "SEMICONDUCTORS", "The reported result remains uncertain."
    ) is None


def test_literal_evidence_and_uncertainty_checks_still_apply() -> None:
    invalid = hypothesis(
        "Reported profitability could alter investor earnings expectations."
    )
    invalid.impacts[0].evidence_excerpt = "Invented result"
    with pytest.raises(ImpactAnalysisError) as exc:
        validate_result(event(), invalid)
    assert exc.value.code == "EVIDENCE_MISMATCH"

    invalid = hypothesis(
        "Reported profitability could alter investor earnings expectations."
    )
    invalid.impacts[0].uncertainty = "LOW"
    with pytest.raises(ImpactAnalysisError) as exc:
        validate_result(event(), invalid)
    assert exc.value.code == "UNCERTAINTY_MISMATCH"


def test_native_ollama_repair_does_not_repeat_rejected_text() -> None:
    from tris_market_event_intelligence.impact import ollama_native_schema

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        assert payload["format"] == ollama_native_schema(event())
        if len(calls) == 1:
            result = hypothesis(
                "Improved financial performance may indicate broader economic strength."
            )
        else:
            result = ImpactResult(impacts=[])
            assert "MARKET_MECHANISM_MISSING" in payload["messages"][-1]["content"]
            assert "Improved financial performance" not in json.dumps(payload["messages"])
        return httpx.Response(
            200, json={"message": {"content": result.model_dump_json()}}
        )

    provider = OpenAICompatibleProvider(
        base_url="http://127.0.0.1:11434/v1", api_key="local-placeholder",
        model="offline-test", native_ollama=True,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert provider.analyze(event()).impacts == []
    assert len(calls) == 2
