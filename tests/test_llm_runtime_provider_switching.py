"""Generic model switching preserves the existing NEWS source/evidence contracts."""
from __future__ import annotations

import json

import httpx
import pytest

from test_gemma4_news_comparison import FakeAnalyzer, event, good_result
from test_local_impact import example_event, sample_impact
from test_news_pressure_provider import (
    event as pressure_event,
    impact as pressure_impact,
    model_reply,
)

from tris_market_event_intelligence import gemma4_news_comparison
from tris_market_event_intelligence.gemma4_news_comparison import (
    compare_news_models,
)
from tris_market_event_intelligence.impact import ImpactResult
from tris_market_event_intelligence.llm_runtime import (
    ConfiguredLLMProvider,
    LLMCapabilities,
    LLMExecutionError,
    LLMSettings,
)
from tris_market_event_intelligence.news_llm_factory import (
    build_impact_provider,
    build_pressure_provider,
    with_model,
)


def local(model: str = "gemma4:e2b", **kwargs) -> LLMSettings:
    return LLMSettings(
        kind="ollama", endpoint="http://127.0.0.1:11434", model_id=model,
        **kwargs,
    )


def test_legacy_name_does_not_auto_switch_remote(monkeypatch) -> None:
    monkeypatch.setenv("MARKET_EVENT_LLM_MODEL", "qwen3:4b")
    monkeypatch.delenv("MARKET_EVENT_LLM_PROVIDER", raising=False)
    cfg = LLMSettings.from_environment()
    assert cfg.kind == "ollama"
    assert cfg.model_id == "qwen3:4b"
    monkeypatch.setenv("MARKET_EVENT_LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("MARKET_EVENT_LLM_API_KEY", "private-token")
    with pytest.raises(ValueError, match="explicit opt-in"):
        LLMSettings.from_environment()
    remote = LLMSettings.from_environment(explicit=True)
    assert remote.kind == "openai-compatible"
    assert remote.model_id == "qwen3:4b"


@pytest.mark.parametrize("endpoint", [
    "http://external.example.com/v1", "https://user:password@example.org/v1",
    "ftp://example.org/v1", "https://example.org/v1?secret=1",
])
def test_remote_url_rejects_unsafe_endpoints(endpoint: str) -> None:
    with pytest.raises(ValueError, match="endpoint|HTTPS"):
        LLMSettings(
            kind="openai-compatible", model_id="provider-model", endpoint=endpoint,
            api_key="private",
        )


def test_local_capabilities_and_model_switch_do_not_modify_legacy_state() -> None:
    cfg = local(
        capabilities=LLMCapabilities(
            json_schema=True, thinking=False,
            context_window=4096, max_output_tokens=2048,
        )
    )
    changed = with_model(cfg, "llama3.2:3b")
    assert cfg.model_id == "gemma4:e2b"
    assert changed.model_id == "llama3.2:3b"


def test_local_ollama_transport_with_capabilities_and_preflight() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:4b"}]})
        payload = json.loads(request.content)
        assert payload["model"] == "qwen3:4b"
        assert payload["options"]["num_ctx"] == 4096
        assert payload["options"]["num_predict"] == 2048
        assert payload["options"]["temperature"] == 0
        assert payload["format"]["type"] == "object"
        assert payload["stream"] is False
        assert "think" not in payload
        return httpx.Response(200, json={
            "message": {"content": '{"ok": true}'}, "done_reason": "stop",
        })

    cfg = local("qwen3:4b", capabilities=LLMCapabilities(
        context_window=4096, max_output_tokens=2048,
    ))
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        runtime = ConfiguredLLMProvider(cfg, client=client)
        assert runtime.ensure_local_installed() is True
        assert json.loads(runtime.generate_json(
            system="no stock", user="original news",
            schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
        )) == {"ok": True}
    assert [r.url.path for r in calls] == ["/api/tags", "/api/chat"]


def test_remote_json_only_transport_never_leaks_key() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        payload = json.loads(request.content)
        assert payload["model"] == "remote-model"
        assert payload["response_format"] == {"type": "json_object"}
        assert request.headers["authorization"] == "Bearer private-key"
        return httpx.Response(200, json={"choices": [
            {"message": {"content": '{"ok": 1}'}},
        ]})

    cfg = LLMSettings(
        kind="openai-compatible", model_id="remote-model",
        endpoint="https://service.example/v1", api_key="private-key",
        capabilities=LLMCapabilities(json_schema=False),
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = ConfiguredLLMProvider(cfg, client=client).generate_json(
            system="source guard", user="sample", schema=None,
        )
    assert result == '{"ok": 1}'
    assert requests[0].url.path == "/v1/chat/completions"
    assert "private-key" not in result


def test_declared_unsupported_json_schema_never_calls_model() -> None:
    calls = []
    with httpx.Client(transport=httpx.MockTransport(
        lambda req: (calls.append(req), httpx.Response(200))[1],
    )) as client:
        runtime = ConfiguredLLMProvider(local(capabilities=LLMCapabilities(
            json_schema=False,
        )), client=client)
        with pytest.raises(LLMExecutionError, match="STRUCTURED_OUTPUT_UNSUPPORTED"):
            runtime.generate_json(
                system="news", user="data", schema={"type": "object"},
            )
    assert not calls


def test_impact_factory_reuses_strict_source_evidence_validator() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        assert json.loads(request.content)["format"]["type"] == "object"
        return httpx.Response(200, json={"message": {"content": ImpactResult(
            impacts=[sample_impact()],
        ).model_dump_json()}})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = build_impact_provider(local("qwen3:4b"), client=client)
        assert provider.model_id == "qwen3:4b"
        assert provider.analyze(example_event()).impacts[0].entities == ["NVIDIA"]


def test_injected_pressure_keeps_unchanged_news_contract() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        assert json.loads(request.content)["format"]["type"] == "object"
        return httpx.Response(200, json={"message": {"content": model_reply()}})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = build_pressure_provider(local("llama3.2:3b"), client=client).analyze(
            pressure_event(), pressure_impact(), generated_at="2026-11-15T01:00:00Z",
        )
    assert result.intensity == "MODERATE"
    assert result.model_id == "llama3.2:3b"
    assert result.market_prices_used is False


def test_same_source_news_comparison_accepts_arbitrary_pair_only_explicitly(
    monkeypatch, tmp_path,
) -> None:
    monkeypatch.setattr(
        gemma4_news_comparison, "select_news_events",
        lambda path, **kwargs: [event()],
    )
    pair = ("qwen3:4b", "llama3.2:3b")
    report = compare_news_models(
        db_path=tmp_path / "not-opened", models=pair,
        analyzers=(FakeAnalyzer(pair[0], "empty"), FakeAnalyzer(pair[1], "valid")),
        limit=1,
    )
    assert report["schemaVersion"] == "news_llm_side_by_side_v1"
    assert report["selectedNewsEvents"] == 1
    assert report["modelCallCount"] == 2
    assert report["independentNewsAccuracyMeasured"] is False
    assert report["productionModelChanged"] is False
    assert report["marketPricesUsed"] is False
    assert report["models"] == list(pair)
    with pytest.raises(ValueError, match="distinct"):
        compare_news_models(
            db_path=tmp_path / "not-opened", models=(pair[0], pair[0]),
        )
