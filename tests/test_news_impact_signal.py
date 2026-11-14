"""News pressure signal contracts never forecast ticker returns."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from tris_market_event_intelligence.news_impact_signal import (
    NewsImpactSignal,
    legacy_impact_to_news_signal,
)


def base() -> dict[str, object]:
    return {
        "event_id": 17, "market": "KR", "sector": "SEMICONDUCTORS",
        "direction": "NEGATIVE", "intensity": "MODERATE",
        "intensity_score": 52.0, "horizon": "SHORT_TERM",
        "uncertainty": "HIGH",
        "rationale": "If chip export restrictions tighten, expected shipments may decline.",
        "evidence_excerpt": "Export restrictions on semiconductors",
        "source_url": "https://example.org/news/17",
        "model_id": "local-research", "prompt_version": "news-impact-v1",
        "generated_at": "2026-11-15T00:00:00+00:00",
    }


def test_news_only_signal_preserves_direction_strength_and_downstream_boundary() -> None:
    signal = NewsImpactSignal.model_validate(base())
    assert signal.direction == "NEGATIVE"
    assert signal.intensity == "MODERATE"
    assert signal.intensity_score == 52
    assert signal.price_prediction_owned_by == "DonghakStockVision"
    assert signal.market_prices_used is False
    assert signal.predicts_stock_return is False
    assert signal.predicts_stock_price is False
    assert signal.calibrated_against_market_prices is False
    assert "predicted_return" not in signal.model_dump()
    assert NewsImpactSignal.model_validate_json(signal.model_dump_json()) == signal


@pytest.mark.parametrize(("change", "message"), [
    ({"intensity": "UNASSESSED", "intensity_score": 30}, "Unassessed"),
    ({"intensity": "HIGH", "intensity_score": None}, "requires"),
    ({"direction": "UNCERTAIN"}, "Uncertain direction"),
    ({"intensity_score": 101}, "less than or equal"),
    ({"intensity_score": -3}, "greater than or equal"),
    ({"market_prices_used": True}, "False"),
    ({"predicts_stock_return": True}, "False"),
    ({"predicts_stock_price": True}, "False"),
    ({"calibrated_against_market_prices": True}, "False"),
    ({"price_prediction_owned_by": "TrisMarketEventIntelligence"}, "DonghakStockVision"),
    ({"predicted_stock_price": 123.0}, "Extra inputs"),
])
def test_market_prediction_and_invalid_news_intensity_are_rejected(
    change: dict[str, object], message: str,
) -> None:
    row = {**base(), **change}
    with pytest.raises(ValidationError, match=message):
        NewsImpactSignal.model_validate(row)


def test_existing_impacts_are_not_given_fabricated_strength() -> None:
    legacy = {
        "market": "US", "sector": "TECH", "direction": "POSITIVE",
        "horizon": "MEDIUM_TERM", "uncertainty": "HIGH",
        "rationale": "If demand grows, future software revenues may increase.",
        "evidence_excerpt": "Software spending has risen",
        "model_id": "gemma2:latest", "prompt_version": "market-impact-v8",
        "generated_at": "2026-11-15T00:00:00+00:00",
    }
    output = legacy_impact_to_news_signal(
        legacy, event_id=11, source_url="https://example.org/article",
    )
    assert output.event_id == 11
    assert output.direction == "POSITIVE"
    assert output.intensity == "UNASSESSED"
    assert output.intensity_score is None
    assert output.predicts_stock_return is False
    assert output.model_id == "gemma2:latest"
