"""Isolated Gemma news strength grading cannot invent market returns."""
from __future__ import annotations

import json

import httpx
import pytest

from tris_market_event_intelligence.impact import (
    EventInput,
    Impact,
    ImpactAnalysisError,
)
from tris_market_event_intelligence.news_pressure_provider import (
    NEWS_PRESSURE_PROMPT_VERSION,
    GemmaNewsPressureProvider,
)

EVIDENCE = "Federal funds rate The Committee raised the interest rate by 25 basis points."


def event() -> EventInput:
    return EventInput(
        1, "MONETARY_POLICY", "VERIFIED", "UP", 25, EVIDENCE,
        "fed_monetary", "https://www.federalreserve.gov/a",
        "2026-11-15T00:00:00+00:00",
    )


def impact(direction: str = "NEGATIVE") -> Impact:
    return Impact(
        market="US", sector="TECH", direction=direction, horizon="SHORT_TERM",
        rationale=(
            "Higher rates may pressure technology valuations through "
            "discount rates, but expectations and repricing are unknown."
        ),
        evidence_excerpt="raised the interest rate by 25 basis points",
        uncertainty="HIGH", drivers=["discount rates"],
    )


def model_reply(
    grade: str = "MODERATE", score: float | None = 52.0,
    excerpt: str | None = "raised the interest rate by 25 basis points",
    reason: str = (
        "Conditional discount-rate pressure may transmit to technology valuations."
    ),
) -> str:
    return json.dumps({
        "intensity": grade, "intensity_score": score,
        "reason": reason, "intensity_evidence_excerpt": excerpt,
    })


def provider(reply: str, *, capture: list | None = None) -> GemmaNewsPressureProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "127.0.0.1"
        assert request.url.path == "/api/chat"
        posted = json.loads(request.content)
        assert posted["model"] == "gemma2:latest"
        assert posted["options"]["temperature"] == 0
        assert posted["messages"][1]["content"]
        assert "event_evidence" in posted["messages"][1]["content"]
        assert "prices" not in posted["messages"][1]["content"]
        if capture is not None:
            capture.append(posted)
        return httpx.Response(200, json={"message": {"content": reply}})
    return GemmaNewsPressureProvider(
        client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def test_local_gemma_returns_conditional_news_pressure_not_stock_target() -> None:
    recorded: list = []
    signal = provider(model_reply(), capture=recorded).analyze(
        event(), impact(), generated_at="2026-11-15T01:00:00+00:00",
    )
    assert len(recorded) == 1
    assert signal.intensity == "MODERATE"
    assert signal.intensity_score == 52
    assert signal.direction == "NEGATIVE"
    assert signal.prompt_version == NEWS_PRESSURE_PROMPT_VERSION
    assert signal.market_prices_used is False
    assert signal.predicts_stock_return is False
    assert signal.predicts_stock_price is False
    assert signal.price_prediction_owned_by == "DonghakStockVision"


@pytest.mark.parametrize("reply", [
    model_reply("UNASSESSED", 32),
    model_reply("HIGH", 30),
    model_reply("MODERATE", None),
    model_reply("LOW", 50),
    model_reply("MODERATE", 52, "invented source excerpt"),
    model_reply("HIGH", 75, None),
    model_reply("HIGH", 105),
    json.dumps({"intensity": "LOW", "intensity_score": 20,
                "reason": "The model predicts a 5% fall."}),
])
def test_invalid_or_unverifiable_strength_is_rejected(reply: str) -> None:
    with pytest.raises(ImpactAnalysisError):
        provider(reply).analyze(
            event(), impact(), generated_at="2026-11-15T01:00:00+00:00",
        )


def test_unassessed_grade_is_valid_without_score_or_evidence_quote() -> None:
    signal = provider(model_reply(
        "UNASSESSED", None, None,
        reason="The report contains too little information to grade news pressure.",
    )).analyze(event(), impact(), generated_at="2026-11-15T01:00:00+00:00")
    assert signal.intensity == "UNASSESSED"
    assert signal.intensity_score is None


def test_uncertain_direction_skips_second_model_call() -> None:
    def no_call(_: httpx.Request) -> httpx.Response:
        raise AssertionError("Should not invoke a model for uncertain direction")

    client = httpx.Client(transport=httpx.MockTransport(no_call))
    signal = GemmaNewsPressureProvider(client=client).analyze(
        event(), impact(direction="UNCERTAIN"),
        generated_at="2026-11-15T01:00:00+00:00",
    )
    assert signal.intensity == "UNASSESSED"
    assert signal.intensity_score is None


def test_original_impact_must_pass_evidence_checks_before_model_call() -> None:
    invalid = impact().model_copy(update={"evidence_excerpt": "fabricated"})
    with pytest.raises(ImpactAnalysisError, match="absent"):
        provider(model_reply()).analyze(
            event(), invalid, generated_at="2026-11-15T01:00:00+00:00",
        )


def test_noneligible_source_never_invokes_model() -> None:
    old = event()
    candidate = EventInput(
        old.event_id, old.classification, "CANDIDATE", old.direction,
        old.change_basis_points, old.evidence, old.source, old.source_url,
        old.detected_at,
    )
    with pytest.raises(ImpactAnalysisError, match="eligible"):
        provider(model_reply()).analyze(
            candidate, impact(), generated_at="2026-11-15T01:00:00+00:00",
        )
