"""Offline target guardrail cases taken from source-only FOMC repeat-run failures."""
from __future__ import annotations

import json

import pytest

from tris_market_event_intelligence.gemma_original_article_batch_cli import (
    AnswerError,
    validate_answer,
)
from tris_market_event_intelligence.news_impact_target_policy import (
    target_error,
    target_key,
)


@pytest.mark.parametrize("label", [
    "Federal Reserve", "Federal Funds Rate",
    "Federal Funds Rate / Primary Credit Rate", "Primary credit rate",
    "Open Market Operations", "Repurchase Agreement Operations",
    "Economic activity", "longer-maturity Treasury yields",
])
def test_policy_variables_and_actors_are_not_impact_targets(label: str) -> None:
    assert target_error(label) == "NON_MARKET_TARGET"
    payload = json.dumps({
        "summary": "Policy news", "limitations": "No outcome assessed",
        "impacts": [{
            "target": label, "direction": "UP",
            "reason": "The policy statement said a rate increased",
            "evidence_id": "E1",
        }],
    })
    with pytest.raises(AnswerError, match="NON_MARKET_TARGET"):
        validate_answer(payload, {"E1": "The central bank issued a statement."})


@pytest.mark.parametrize("label", [
    "Financial Markets", "General market", "Treasury securities",
    "Corporate bonds issued by AI hyperscalers", "Banking System",
    "Technology", "NVIDIA", "US Equity Index",
])
def test_market_exposure_targets_not_auto_rejected(label: str) -> None:
    assert target_error(label) is None


def test_alias_key_preserves_original_named_target() -> None:
    answer = validate_answer(json.dumps({
        "summary": "Policy decision", "limitations": "No price comparison",
        "impacts": [{
            "target": "General market", "direction": "UNCERTAIN",
            "reason": "Broad cross-market mechanism cannot be determined",
            "evidence_id": "E1",
        }],
    }), {"E1": "The committee announced an updated meeting schedule."})
    impact = answer["impacts"][0]
    assert impact["target"] == "General market"
    assert impact["target_key"] == target_key("Financial Markets")
    assert impact["evidence_quote"] == (
        "The committee announced an updated meeting schedule."
    )


def test_invalid_target_alongside_valid_target_fails_whole_model_response() -> None:
    payload = json.dumps({
        "summary": "Rates", "limitations": "N/A",
        "impacts": [
            {"target": "Banking System", "direction": "UNCERTAIN",
             "reason": "Funding costs may change", "evidence_id": "E1"},
            {"target": "Federal Reserve", "direction": "UP",
             "reason": "An institution is not an asset", "evidence_id": "E1"},
        ],
    })
    with pytest.raises(AnswerError, match="NON_MARKET_TARGET"):
        validate_answer(payload, {"E1": "The committee approved an action."})
