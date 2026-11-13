"""Regression for first real Gemma 2 failure: short rationale in local output."""

import json

import httpx
import pytest

from tris_market_event_intelligence.impact import (
    PROMPT_VERSION,
    EventInput,
    ImpactAnalysisError,
    OpenAICompatibleProvider,
    ollama_native_schema,
)


def event() -> EventInput:
    return EventInput(
        6, "ECONOMIC_INDICATOR", "UNVERIFIED", None, None,
        "Consumer Price Index increased this month.",
        "bbc_business", "https://example.org/news", "now", "REPORTED",
    )


def hypothesis(rationale: str) -> str:
    return json.dumps({"impacts": [{
        "market": "US", "sector": "BROAD_MARKET", "direction": "UNCERTAIN",
        "horizon": "SHORT_TERM", "rationale": rationale,
        "evidence_excerpt": "Price Index increased",
        "uncertainty": "HIGH", "drivers": ["reported inflation"],
        "entities": [],
    }]})


def provider(handler) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        base_url="http://127.0.0.1:11434/v1",
        api_key="local-placeholder", model="gemma2:latest",
        native_ollama=True,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_native_schema_includes_all_pydantic_length_bounds() -> None:
    assert PROMPT_VERSION == "market-impact-v6"
    result = ollama_native_schema(event())
    assert result["properties"]["impacts"]["maxItems"] == 12
    item = result["properties"]["impacts"]["items"]
    rationale = item["properties"]["rationale"]
    assert rationale["minLength"] == 10
    assert rationale["maxLength"] == 2000
    quote = item["properties"]["evidence_excerpt"]
    assert quote["minLength"] == 8
    assert quote["maxLength"] == 1000
    assert item["properties"]["drivers"]["minItems"] == 1
    assert item["properties"]["drivers"]["maxItems"] == 6
    assert item["properties"]["entities"]["maxItems"] == 8


def test_short_rationale_is_retried_once_then_verified() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if len(calls) == 1:
            assert body["format"]["properties"]["impacts"]["items"]["properties"][
                "rationale"
            ]["minLength"] == 10
            return httpx.Response(200, json={
                "message": {"content": hypothesis("short")},
            })
        assert len(calls) == 2
        assert "SCHEMA_STRING_TOO_SHORT_RATIONALE" in body["messages"][-1]["content"]
        assert "short" not in body["messages"][-1]["content"]
        return httpx.Response(200, json={
            "message": {"content": hypothesis(
                "Reported higher inflation could affect valuations if confirmed."
            )},
        })

    output = provider(handler).analyze(event())
    assert len(output.impacts) == 1
    assert len(output.impacts[0].rationale) >= 10
    assert len(calls) == 2


def test_two_invalid_rationales_fail_without_silent_repair() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={
            "message": {"content": hypothesis("tiny")},
        })

    with pytest.raises(ImpactAnalysisError) as error:
        provider(handler).analyze(event())
    assert error.value.code == "SCHEMA_STRING_TOO_SHORT_RATIONALE"
    assert len(calls) == 2


def test_http_503_never_retries_or_downgrades() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(503, json={"error": "unavailable"})

    with pytest.raises(httpx.HTTPStatusError):
        provider(handler).analyze(event())
    assert len(calls) == 1


def test_blank_impact_output_is_valid_no_hypothesis() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": '{"impacts":[]}'}})

    assert provider(handler).analyze(event()).impacts == []


def test_evidence_mismatch_still_cannot_be_saved_as_hypothesis() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={
            "message": {"content": json.dumps({"impacts": [{
                "market": "US", "sector": "BROAD_MARKET",
                "direction": "UNCERTAIN", "horizon": "SHORT_TERM",
                "rationale": "This fabricated impact claim must be rejected.",
                "evidence_excerpt": "invented sentence absent from evidence",
                "uncertainty": "HIGH", "drivers": ["unverified data"],
                "entities": [],
            }]})},
        })

    with pytest.raises(ImpactAnalysisError) as error:
        provider(handler).analyze(event())
    assert error.value.code == "EVIDENCE_MISMATCH"
    assert len(calls) == 2
