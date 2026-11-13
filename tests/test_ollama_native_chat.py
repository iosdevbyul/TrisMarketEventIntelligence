"""One bounded native Ollama request with compatibility handling."""

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
    ollama_native_schema,
)


def sample_event() -> EventInput:
    return EventInput(
        6, "ECONOMIC_INDICATOR", "UNVERIFIED", None, None,
        "Consumer Price Index increased.", "bbc_business",
        "https://example.org/news", "now", "REPORTED",
    )


def good_result() -> str:
    return ImpactResult(impacts=[Impact(
        market="US", sector="BROAD_MARKET", direction="UNCERTAIN",
        horizon="SHORT_TERM",
        rationale="A reported inflation increase could affect valuations if verified.",
        evidence_excerpt="Price Index increased", uncertainty="HIGH",
        drivers=["unverified price change"], entities=[],
    )]).model_dump_json()


def test_native_schema_is_flat_and_restricts_high() -> None:
    assert PROMPT_VERSION == "market-impact-v6"
    schema = ollama_native_schema(sample_event())
    assert "$defs" not in json.dumps(schema)
    item = schema["properties"]["impacts"]["items"]
    assert item["properties"]["uncertainty"]["enum"] == ["HIGH"]
    assert item["required"] == list(item["properties"])


def test_native_chat_parses_response_and_uses_format() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = json.loads(request.content)
        assert body["model"] == "gemma2:latest"
        assert body["stream"] is False
        assert body["format"]["properties"]["impacts"]["items"]["properties"][
            "uncertainty"
        ]["enum"] == ["HIGH"]
        assert body["options"]["temperature"] == 0
        return httpx.Response(200, json={"message": {"content": good_result()}})

    provider = OpenAICompatibleProvider(
        base_url="http://127.0.0.1:11434/v1", api_key="placeholder",
        model="gemma2:latest", native_ollama=True,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert provider.analyze(sample_event()).impacts[0].uncertainty == "HIGH"


def test_400_falls_back_once_to_json_and_keeps_validator() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload["format"])
        if len(calls) == 1:
            return httpx.Response(400, json={"error": "unsupported schema"})
        return httpx.Response(200, json={"message": {"content": good_result()}})

    provider = OpenAICompatibleProvider(
        base_url="http://127.0.0.1:11434/v1", api_key="placeholder",
        model="gemma2:latest", native_ollama=True,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert provider.analyze(sample_event()).impacts
    assert len(calls) == 2
    assert calls[1] == "json"


def test_fallback_does_not_accept_lower_risk() -> None:
    bad = json.loads(good_result())
    bad["impacts"][0]["uncertainty"] = "LOW"
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(422, json={"error": "unsupported"})
        return httpx.Response(200, json={"message": {"content": json.dumps(bad)}})

    provider = OpenAICompatibleProvider(
        base_url="http://127.0.0.1:11434/v1", api_key="placeholder",
        model="gemma2:latest", native_ollama=True,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ImpactAnalysisError) as error:
        provider.analyze(sample_event())
    assert error.value.code == "UNCERTAINTY_MISMATCH"
    assert calls == 3


def test_other_http_errors_never_fallback() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503, json={"error": "unavailable"})

    provider = OpenAICompatibleProvider(
        base_url="http://127.0.0.1:11434/v1", api_key="placeholder",
        model="gemma2:latest", native_ollama=True,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(httpx.HTTPStatusError):
        provider.analyze(sample_event())
    assert calls == 1
