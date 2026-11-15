"""Versioned NEWS-only potential-impact magnitude research contract.

Magnitude means a reasoned hypothetical *organizational/material* impact
of a reported event. It is NOT the direction or size of stock movements.
This module neither retrieves market prices nor creates human gold.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

BANDS = ("LOW", "MODERATE", "HIGH", "CRITICAL")
FACT_STATUSES = ("ALLEGED", "REPORTED", "CONFIRMED", "DISPUTED")
DIRECTIONS = ("POSITIVE", "NEGATIVE", "MIXED", "UNCERTAIN")


class NewsMaterialityHypothesis(BaseModel):
    """One provisional claim about how consequential a NEWS event could be."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["news_materiality_hypothesis_v1"] = (
        "news_materiality_hypothesis_v1"
    )
    article_id: int = Field(gt=0)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    entity: str = Field(min_length=2, max_length=120)
    event_summary: str = Field(min_length=10, max_length=500)
    fact_status: Literal["ALLEGED", "REPORTED", "CONFIRMED", "DISPUTED"]
    potential_magnitude: Literal[
        "LOW", "MODERATE", "HIGH", "CRITICAL", "UNASSESSED"
    ]
    direction: Literal["POSITIVE", "NEGATIVE", "MIXED", "UNCERTAIN"]
    evidence_excerpt: str = Field(min_length=8, max_length=1000)
    conditional_mechanism: str = Field(min_length=20, max_length=1500)
    verification_needed: tuple[str, ...] = Field(max_length=8)
    magnitude_basis: Literal[
        "HUMAN_OR_MODEL_HYPOTHESIS_UNCALIBRATED", "NOT_ASSESSED"
    ]
    market_prices_used: Literal[False] = False
    price_prediction: Literal[False] = False
    independent_human_gold: Literal[False] = False

    @model_validator(mode="after")
    def enforce_claim_separation(self) -> NewsMaterialityHypothesis:
        if self.potential_magnitude == "UNASSESSED":
            if self.magnitude_basis != "NOT_ASSESSED":
                raise ValueError("Unassessed magnitude must be marked NOT_ASSESSED")
        elif self.magnitude_basis != "HUMAN_OR_MODEL_HYPOTHESIS_UNCALIBRATED":
            raise ValueError("Assessed magnitude must be explicitly uncalibrated")
        # Direction and magnitude are orthogonal. An important leadership
        # change can have a large impact even if its direction is unknown.
        return self
