"""Quality gate regressions: obvious truncation versus semantic uncertainty."""

import pytest

from tris_market_event_intelligence.impact import (
    EventInput,
    Impact,
    ImpactAnalysisError,
    ImpactResult,
    validate_result,
)
from tris_market_event_intelligence.impact_quality import rationale_quality_code


def test_observed_korean_truncation_is_rejected() -> None:
    event = EventInput(
        6, "INDUSTRY", "UNVERIFIED", None, None,
        "반도체 경기 호조의 영향으로 수출과 투자의 높은 증가세가 지속되고",
        "sample", "https://example.org/a", "now", "REPORTED",
    )
    result = ImpactResult(impacts=[Impact(
        market="GLOBAL", sector="SEMICONDUCTORS", direction="POSITIVE",
        horizon="MEDIUM_TERM",
        rationale="반도체 경기 호조의 영향으로 수출과 투자의 높은 증가세가 지속되고",
        evidence_excerpt="반도체 경기 호조의 영향으로 수출과 투자의 높은 증가세",
        uncertainty="HIGH", drivers=[], entities=["반도체 경기 호조"],
    )])
    with pytest.raises(ImpactAnalysisError) as error:
        validate_result(event, result)
    assert error.value.code == "RATIONALE_INCOMPLETE"


@pytest.mark.parametrize("text", [
    "Reported sector demand could increase, but",
    "The effect depends on",
    "여러 산업에 파급효과가 나타나고.",
])
def test_clear_dangling_connectives(text: str) -> None:
    assert rationale_quality_code(text) == "RATIONALE_INCOMPLETE"


@pytest.mark.parametrize("text", [
    "If confirmed, reported price increases might pressure valuations.",
    "확인되지 않은 반도체 보도이므로 시장 반응은 불확실하다.",
    "The evidence may be insufficient to infer a directional effect",
])
def test_complete_or_uncertain_rationale_not_rejected(text: str) -> None:
    assert rationale_quality_code(text) is None


def test_empty_hypotheses_are_accepted() -> None:
    event = EventInput(
        6, "INDUSTRY", "UNVERIFIED", None, None,
        "반도체 경기 호조", "sample", "https://example.org/a", "now", "REPORTED",
    )
    assert validate_result(event, ImpactResult(impacts=[])).impacts == []


def test_quality_gate_does_not_claim_semantic_verification() -> None:
    # This syntactically complete but unsupported causal leap is NOT proved
    # correct by a deterministic ending check; it requires human review.
    assert rationale_quality_code(
        "The reported result will definitely cause global stocks to rise."
    ) is None
