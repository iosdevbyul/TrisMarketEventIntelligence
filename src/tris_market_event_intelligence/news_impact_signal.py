"""News-only impact signals for downstream price forecasting ownership.

Strength measures *hypothesized news pressure*, not expected stock returns.
The legacy impact model does not estimate strength; converting legacy rows
must therefore mark strength as UNASSESSED instead of inventing a score.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class NewsImpactSignal(BaseModel):
    """Strict downstream contract, separate from stock-price forecasting."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["news_impact_signal_v1"] = "news_impact_signal_v1"
    event_id: int = Field(gt=0)
    market: Literal["US", "KR", "GLOBAL"]
    sector: str = Field(min_length=1, max_length=80)
    direction: Literal["POSITIVE", "NEGATIVE", "MIXED", "UNCERTAIN"]
    intensity: Literal["LOW", "MODERATE", "HIGH", "UNASSESSED"]
    intensity_score: float | None = Field(default=None, ge=0, le=100)
    horizon: Literal["SHORT_TERM", "MEDIUM_TERM"]
    uncertainty: Literal["LOW", "MEDIUM", "HIGH"]
    rationale: str = Field(min_length=10, max_length=2000)
    evidence_excerpt: str = Field(min_length=8, max_length=1000)
    source_url: str = Field(min_length=8, max_length=1000)
    model_id: str = Field(min_length=1, max_length=120)
    prompt_version: str = Field(min_length=1, max_length=80)
    generated_at: str = Field(min_length=10, max_length=64)
    price_prediction_owned_by: Literal["DonghakStockVision"] = "DonghakStockVision"
    market_prices_used: Literal[False] = False
    predicts_stock_return: Literal[False] = False
    predicts_stock_price: Literal[False] = False
    calibrated_against_market_prices: Literal[False] = False

    @model_validator(mode="after")
    def validate_intensity(self) -> NewsImpactSignal:
        if self.intensity == "UNASSESSED" and self.intensity_score is not None:
            raise ValueError("Unassessed signal cannot have a strength score")
        if self.intensity != "UNASSESSED" and self.intensity_score is None:
            raise ValueError("Assessed signal requires an explicit intensity score")
        if self.direction == "UNCERTAIN" and self.intensity != "UNASSESSED":
            raise ValueError("Uncertain direction cannot assert directional strength")
        return self


def legacy_impact_to_news_signal(
    impact: dict[str, object], *, event_id: int, source_url: str,
) -> NewsImpactSignal:
    """Preserve existing AI output without fabricating news-impact strength.

    Accepts the persisted market_impacts fields (not proposed stock prices).
    An actual strength estimate requires a separately validated model update.
    """
    return NewsImpactSignal(
        event_id=event_id,
        market=impact["market"],
        sector=impact["sector"],
        direction=impact["direction"],
        intensity="UNASSESSED",
        intensity_score=None,
        horizon=impact["horizon"],
        uncertainty=impact["uncertainty"],
        rationale=impact["rationale"],
        evidence_excerpt=impact["evidence_excerpt"],
        source_url=source_url,
        model_id=impact["model_id"],
        prompt_version=impact["prompt_version"],
        generated_at=impact["generated_at"],
    )
