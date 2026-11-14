"""Independent triage regressions; these do not claim a fact-checker exists."""
import pytest

from tris_market_event_intelligence.impact_semantic_review import semantic_review_packet


def packet(source: str, excerpt: str, rationale: str) -> dict[str, object]:
    return semantic_review_packet(
        source_text=source, evidence_excerpt=excerpt, rationale=rationale,
    )


def test_quote_verbatim_is_never_semantic_approval() -> None:
    result = packet(
        "The company increased revenue in Q2.",
        "increased revenue in Q2",
        "The report guarantees an immediate rise in the company's share price.",
    )
    assert result["sourceExcerptMatches"] is True
    assert result["status"] == "REQUIRES_HUMAN_REVIEW"
    assert result["semanticallyVerified"] is False
    assert "RATIONALE_CERTAINTY_REVIEW" in result["reviewCodes"]


def test_negation_outside_selected_quote_is_flagged() -> None:
    result = packet(
        "The company did not confirm the acquisition.",
        "confirm the acquisition",
        "The acquisition is expected to improve cash flow.",
    )
    assert "NEGATION_OUTSIDE_EXCERPT_REVIEW" in result["reviewCodes"]
    assert "SOURCE_NEGATION_OMITTED_REVIEW" in result["reviewCodes"]


def test_conditional_reporting_is_not_promoted_to_certainty() -> None:
    result = packet(
        "If approved, the company could increase capacity.",
        "the company could increase capacity",
        "The capacity expansion will inevitably lift profits.",
    )
    assert "CONDITION_OUTSIDE_EXCERPT_REVIEW" in result["reviewCodes"]
    assert "RATIONALE_CERTAINTY_REVIEW" in result["reviewCodes"]


def test_past_time_frame_is_visible_to_reviewer() -> None:
    result = packet(
        "Last year, the company reported higher exports.",
        "the company reported higher exports",
        "Higher exports increase current earnings expectations.",
    )
    assert "TIME_OUTSIDE_EXCERPT_REVIEW" in result["reviewCodes"]
    assert "SOURCE_TIME_OMITTED_REVIEW" in result["reviewCodes"]


def test_korean_negation_and_conditional_context() -> None:
    result = packet(
        "당국은 인수를 승인하지 않았으며 향후 승인 가능성을 검토한다고 밝혔다.",
        "인수를 승인",
        "인수가 확정됐다. 관련 주가 상승이 반드시 발생한다.",
    )
    assert "NEGATION_OUTSIDE_EXCERPT_REVIEW" in result["reviewCodes"]
    assert "SOURCE_NEGATION_OMITTED_REVIEW" in result["reviewCodes"]
    assert "RATIONALE_CERTAINTY_REVIEW" in result["reviewCodes"]


def test_no_keyword_signal_still_needs_human_semantic_check() -> None:
    result = packet(
        "The firm reported a quarterly sales increase.",
        "quarterly sales increase",
        "A sales increase could affect earnings expectations.",
    )
    assert result["reviewCodes"] == []
    assert result["status"] == "REQUIRES_HUMAN_REVIEW"


def test_unmatched_quote_and_invalid_inputs() -> None:
    result = packet("The board denied the deal.", "approved the deal", "A deal happened.")
    assert result["reviewCodes"][0] == "EXCERPT_NOT_IN_SOURCE"
    assert result["sourceExcerptMatches"] is False
    with pytest.raises(ValueError):
        packet("", "a quote", "a rationale")
    with pytest.raises(TypeError):
        semantic_review_packet(source_text=None, evidence_excerpt="a", rationale="b")
