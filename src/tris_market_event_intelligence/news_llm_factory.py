"""Explicit LLM dependency injection for existing, unchanged NEWS contracts."""
from __future__ import annotations

from dataclasses import replace

import httpx

from tris_market_event_intelligence.impact import OpenAICompatibleProvider
from tris_market_event_intelligence.llm_runtime import (
    ConfiguredLLMProvider,
    LLMSettings,
)
from tris_market_event_intelligence.news_pressure_provider import (
    GemmaNewsPressureProvider,
)


def build_runtime(
    settings: LLMSettings, *, client: httpx.Client | None = None,
    confirm_model: bool = False,
) -> ConfiguredLLMProvider:
    runtime = ConfiguredLLMProvider(settings, client=client)
    if settings.kind == "ollama" and confirm_model and not runtime.ensure_local_installed():
        raise ValueError("Explicitly requested Ollama model is not installed")
    return runtime


def build_impact_provider(
    settings: LLMSettings, *, client: httpx.Client | None = None,
    confirm_model: bool = False,
) -> OpenAICompatibleProvider:
    """Reuse the original ImpactProvider, including its news validator."""
    runtime = build_runtime(settings, client=client, confirm_model=confirm_model)
    return OpenAICompatibleProvider(
        base_url=(settings.endpoint.rstrip("/") + "/v1"
                  if settings.kind == "ollama" else settings.endpoint),
        api_key=(settings.api_key if settings.kind != "ollama" else "local-placeholder"),
        model=settings.model_id,
        timeout_seconds=settings.timeout_seconds,
        structured_output=settings.capabilities.json_schema,
        runtime=runtime,
    )


def build_pressure_provider(
    settings: LLMSettings, *, client: httpx.Client | None = None,
    confirm_model: bool = False,
) -> GemmaNewsPressureProvider:
    """Reuse the pressure analyzer and every existing evidence guardrail."""
    if not settings.capabilities.json_schema:
        raise ValueError("Pressure requires JSON Schema capability")
    runtime = build_runtime(settings, client=client, confirm_model=confirm_model)
    return GemmaNewsPressureProvider(
        model=settings.model_id, timeout_seconds=settings.timeout_seconds,
        runtime=runtime,
    )


def with_model(settings: LLMSettings, model_id: str) -> LLMSettings:
    """Explicitly choose a separate model without mutating ambient state."""
    return replace(settings, model_id=model_id)
