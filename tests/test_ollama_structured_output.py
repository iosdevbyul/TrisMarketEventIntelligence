"""Local Ollama schema response formatting with strict post-validation."""

import json

import httpx
import pytest

from tris_market_event_intelligence.impact import (
    PROMPT_VERSION,
    EventInput,
    Impact,
    ImpactAnalysisError,
    ImpactResult,
    OpenAICompatibleProvider,
    structured_impact_schema,
)


def event(status: str = "UNVERIFIED") -> EventInput:
    return EventInput(
        6, "ECONOMIC_INDICATOR", status, None, None,
        "Consumer Price Index increased this month.",
        "bbc_business", "https://example.org/news", "now", "REPORTED",
    )


def sample() -> ImpactResult:
    return ImpactResult(impacts=[Impact(
        market="US", sector="BROAD_MARKET", direction="UNCERTAIN",
        horizon="SHORT_TERM",
        rationale="This reported price movement may affect valuations if verified.",
        evidence_excerpt="Price Index increased", uncertainty="HIGH",
        drivers=["reported inflation"], entities=[],
    )])


def test_schema_is_versioned_and_source_specific() -> None:
    assert PROMPT_VERSION == "market-impact-v6"
    unknown = structured_impact_schema(event())
    known = structured_impact_schema(event("VERIFIED"))
    props = unknown["$defs"]["Impact"]["properties"]
    assert props["uncertainty"]["enum"] == ["HIGH"]
    assert known["$defs"]["Impact"]["properties"]["uncertainty"]["enum"] == [
        "LOW", "MEDIUM", "HIGH",
    ]
    assert props["market"]["enum"] == ["US", "KR", "GLOBAL"]
    assert unknown["additionalProperties"] is False
    assert unknown["$defs"]["Impact"]["additionalProperties"] is False


def test_local_request_schema_and_result() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        fmt = payload["response_format"]
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["strict"] is True
        assert fmt["json_schema"]["schema"]["$defs"]["Impact"]["properties"][
            "uncertainty"
        ]["enum"] == ["HIGH"]
        return httpx.Response(200, json={
            "choices": [{"message": {"content": sample().model_dump_json()}}],
        })

    provider = OpenAICompatibleProvider(
        base_url="http://127.0.0.1:11434/v1",
        api_key="local-placeholder", model="gemma2:latest",
        structured_output=True,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert provider.analyze(event()).impacts[0].uncertainty == "HIGH"


def test_invalid_schema_has_safe_field_code() -> None:
    provider = OpenAICompatibleProvider(
        base_url="http://127.0.0.1:11434/v1",
        api_key="local-placeholder", model="gemma2:latest",
        structured_output=True,
        client=httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={
                "choices": [{"message": {"content": '{"impacts":[{"market":"KR"}]}'}}],
            })
        )),
    )
    with pytest.raises(ImpactAnalysisError) as error:
        provider.analyze(event())
    assert error.value.code.startswith("SCHEMA_MISSING_")
    assert "KR" not in error.value.code


def test_evidence_guard_still_rejects_fabricated_quote() -> None:
    value = sample()
    value.impacts[0].evidence_excerpt = "fabricated economic numbers"
    provider = OpenAICompatibleProvider(
        base_url="http://127.0.0.1:11434/v1",
        api_key="local-placeholder", model="gemma2:latest",
        structured_output=True,
        client=httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={
                "choices": [{"message": {"content": value.model_dump_json()}}],
            })
        )),
    )
    with pytest.raises(ImpactAnalysisError) as error:
        provider.analyze(event())
    assert error.value.code == "EVIDENCE_MISMATCH"


def test_legacy_json_object_response_unchanged() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["response_format"] == {"type": "json_object"}
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"impacts":[]}'}}],
        })

    provider = OpenAICompatibleProvider(
        base_url="https://example.org/v1", api_key="secret", model="remote",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert provider.analyze(event()).impacts == []
