"""Source-aware risk instructions preserve strict validation."""

import json

import httpx
import pytest

from tris_market_event_intelligence.impact import (
    EventInput,
    Impact,
    ImpactAnalysisError,
    ImpactResult,
    OpenAICompatibleProvider,
    PROMPT_VERSION,
    event_specific_instructions,
)


def sample_event(status: str = "UNVERIFIED") -> EventInput:
    return EventInput(
        6, "ECONOMIC_INDICATOR", status, None, None,
        "Prices increased in the latest report.",
        "bbc_business", "https://example.org/report", "now", "REPORTED",
    )


def sample_result(level: str) -> ImpactResult:
    return ImpactResult(impacts=[Impact(
        market="US", sector="BROAD_MARKET", direction="UNCERTAIN",
        horizon="SHORT_TERM",
        rationale="Reported inflation may affect markets if confirmed.",
        evidence_excerpt="Prices increased", uncertainty=level,
        drivers=["reported price change"], entities=[],
    )])


def test_prompt_version_and_source_risk_directive() -> None:
    assert PROMPT_VERSION == "market-impact-v3"
    prompt = event_specific_instructions(sample_event())
    assert "MUST contain uncertainty set to HIGH" in prompt
    assert "LOW and MEDIUM are forbidden" in prompt
    assert "not independently fact checked" in event_specific_instructions(
        sample_event("VERIFIED")
    )


def test_provider_request_contains_trusted_risk_constraint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["messages"][1]["role"] == "system"
        assert "HIGH" in payload["messages"][1]["content"]
        user_data = json.loads(payload["messages"][2]["content"])
        assert user_data["required_uncertainty"] == "HIGH"
        return httpx.Response(200, json={"choices": [{"message": {
            "content": sample_result("HIGH").model_dump_json(),
        }}]})

    provider = OpenAICompatibleProvider(
        base_url="http://127.0.0.1:11434/v1",
        api_key="local-placeholder", model="gemma2:latest",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert provider.analyze(sample_event()).impacts[0].uncertainty == "HIGH"


def test_noncompliant_output_still_fails_instead_of_being_rewritten() -> None:
    provider = OpenAICompatibleProvider(
        base_url="http://127.0.0.1:11434/v1",
        api_key="local-placeholder", model="gemma2:latest",
        client=httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"choices": [{"message": {
                "content": sample_result("LOW").model_dump_json(),
            }}]})
        )),
    )
    with pytest.raises(ImpactAnalysisError) as error:
        provider.analyze(sample_event())
    assert error.value.code == "UNCERTAINTY_MISMATCH"


def test_empty_scenario_remains_valid() -> None:
    provider = OpenAICompatibleProvider(
        base_url="http://127.0.0.1:11434/v1",
        api_key="local-placeholder", model="gemma2:latest",
        client=httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"choices": [{"message": {
                "content": json.dumps({"impacts": []}),
            }}]})
        )),
    )
    assert provider.analyze(sample_event()).impacts == []
