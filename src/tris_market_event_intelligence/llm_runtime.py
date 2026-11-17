"""Opt-in model execution and capability contract shared by NEWS workflows.

The existing ImpactProvider and persisted news contracts remain authoritative.
No model is downloaded, activated, or selected by this module on import.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Literal, Protocol
from urllib.parse import urlsplit

import httpx

ProviderKind = Literal["ollama", "openai-compatible"]
_MODEL = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,119}$")


class LLMExecutionError(ValueError):
    """Stable failure category; does not include raw news or model output."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class LLMCapabilities:
    json_schema: bool = True
    thinking: bool = False
    context_window: int | None = None
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in ("context_window", "max_output_tokens"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or not 128 <= value <= 131072):
                raise ValueError("Invalid configured LLM token resource bound")


@dataclass(frozen=True)
class LLMSettings:
    kind: ProviderKind
    model_id: str
    endpoint: str
    api_key: str = ""
    timeout_seconds: int = 300
    temperature: float = 0
    capabilities: LLMCapabilities = LLMCapabilities()

    def __post_init__(self) -> None:
        if self.kind not in {"ollama", "openai-compatible"}:
            raise ValueError("Unsupported LLM provider kind")
        if not _MODEL.fullmatch(self.model_id):
            raise ValueError("Invalid explicit model identifier")
        parsed = urlsplit(self.endpoint)
        if (not parsed.hostname or parsed.username or parsed.password or
                parsed.query or parsed.fragment or not parsed.netloc):
            raise ValueError("Invalid LLM endpoint")
        loopback = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        if self.kind == "ollama":
            if parsed.scheme != "http" or not loopback or parsed.path not in {"", "/"}:
                raise ValueError("Ollama must use an explicit loopback base URL")
        elif parsed.scheme != "https" and not (
            parsed.scheme == "http" and loopback
        ):
            raise ValueError("Remote LLM endpoint requires HTTPS or loopback")
        if self.kind == "openai-compatible" and not self.api_key:
            raise ValueError("Remote model requires a credential")
        if type(self.timeout_seconds) is not int or not 1 <= self.timeout_seconds <= 900:
            raise ValueError("LLM timeout must be 1..900 seconds")
        if (type(self.temperature) not in {int, float}
                or not 0 <= self.temperature <= 2):
            raise ValueError("LLM temperature must be between zero and two")

    @classmethod
    def from_environment(cls, *, explicit: bool = False) -> LLMSettings:
        """Legacy model name works; no implicit provider switch or remote URL."""
        model = os.getenv("MARKET_EVENT_LLM_MODEL", "gemma2:latest")
        kind = os.getenv("MARKET_EVENT_LLM_PROVIDER", "ollama")
        if kind not in {"ollama", "openai-compatible"}:
            raise ValueError("Unknown MARKET_EVENT_LLM_PROVIDER")
        if kind != "ollama" and not explicit:
            raise ValueError("Remote model selection requires explicit opt-in")
        default = ("http://127.0.0.1:11434" if kind == "ollama"
                   else "https://api.openai.com/v1")
        endpoint = os.getenv("MARKET_EVENT_LLM_BASE_URL", default)
        return cls(
            kind=kind, model_id=model, endpoint=endpoint,
            api_key=(os.getenv("MARKET_EVENT_LLM_API_KEY", "") if kind != "ollama" else ""),
            timeout_seconds=int(os.getenv(
                "MARKET_EVENT_LLM_TIMEOUT_SECONDS",
                "300" if kind == "ollama" else "90",
            )),
            temperature=float(os.getenv("MARKET_EVENT_LLM_TEMPERATURE", "0")),
            capabilities=LLMCapabilities(
                json_schema=os.getenv("MARKET_EVENT_LLM_JSON_SCHEMA", "1") == "1",
                thinking=os.getenv("MARKET_EVENT_LLM_THINKING", "0") == "1",
                context_window=_optional_int("MARKET_EVENT_LLM_CONTEXT_WINDOW"),
                max_output_tokens=_optional_int("MARKET_EVENT_LLM_MAX_OUTPUT_TOKENS"),
            ),
        )


def _optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    return int(raw) if raw else None


class LLMProvider(Protocol):
    @property
    def model_id(self) -> str: ...

    def generate_json(
        self, *, system: str, user: str, schema: dict[str, object] | None,
    ) -> str: ...


class ConfiguredLLMProvider:
    """JSON text transport only. All NEWS meaning and evidence validation stays outside."""

    def __init__(self, settings: LLMSettings, *,
                 client: httpx.Client | None = None) -> None:
        self.settings = settings
        self.client = client
        self._last_generation_diagnostics: dict[str, object] | None = None

    @property
    def last_generation_diagnostics(self) -> dict[str, object] | None:
        # No raw model text, prompts, credentials, or network identifiers.
        return (dict(self._last_generation_diagnostics)
                if self._last_generation_diagnostics is not None else None)

    @property
    def model_id(self) -> str:
        return self.settings.model_id

    def ensure_local_installed(self) -> bool:
        """Read-only local Ollama inventory. Never pulls models or starts services."""
        if self.settings.kind != "ollama":
            raise ValueError("Local preflight is only for Ollama")
        owned = self.client is None
        session = self.client or httpx.Client(
            timeout=httpx.Timeout(self.settings.timeout_seconds), trust_env=False,
        )
        try:
            response = session.get(self.settings.endpoint.rstrip("/") + "/api/tags")
            response.raise_for_status()
            inventory = response.json()
            if not isinstance(inventory, dict) or not isinstance(
                inventory.get("models"), list
            ):
                raise LLMExecutionError("MODEL_INVENTORY_INVALID")
            return any(
                isinstance(row, dict) and row.get("name") == self.model_id
                for row in inventory["models"]
            )
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            if isinstance(exc, LLMExecutionError):
                raise
            raise LLMExecutionError("MODEL_PREFLIGHT_FAILED") from exc
        finally:
            if owned:
                session.close()

    def generate_json(
        self, *, system: str, user: str, schema: dict[str, object] | None,
    ) -> str:
        cfg = self.settings
        self._last_generation_diagnostics = None
        if schema is not None and not cfg.capabilities.json_schema:
            raise LLMExecutionError("STRUCTURED_OUTPUT_UNSUPPORTED")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if cfg.kind == "ollama":
            options: dict[str, int | float] = {"temperature": cfg.temperature}
            if cfg.capabilities.context_window is not None:
                options["num_ctx"] = cfg.capabilities.context_window
            if cfg.capabilities.max_output_tokens is not None:
                options["num_predict"] = cfg.capabilities.max_output_tokens
            payload: dict[str, object] = {
                "model": cfg.model_id, "stream": False,
                "format": schema if schema is not None else "json",
                "messages": messages, "options": options,
            }
            if cfg.capabilities.thinking:
                payload["think"] = True
            endpoint = cfg.endpoint.rstrip("/") + "/api/chat"
            headers: dict[str, str] = {}
        else:
            payload = {
                "model": cfg.model_id, "messages": messages,
                "temperature": cfg.temperature,
                "response_format": (
                    {"type": "json_schema", "json_schema": {
                        "name": "news_analysis", "strict": True, "schema": schema,
                    }} if schema is not None else {"type": "json_object"}
                ),
            }
            if cfg.capabilities.max_output_tokens is not None:
                payload["max_completion_tokens"] = cfg.capabilities.max_output_tokens
            endpoint = cfg.endpoint.rstrip("/") + "/chat/completions"
            headers = {"Authorization": "Bearer " + cfg.api_key}
        owned = self.client is None
        session = self.client or httpx.Client(
            timeout=httpx.Timeout(cfg.timeout_seconds), trust_env=False,
        )
        try:
            response = session.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            if cfg.kind == "ollama":
                self._last_generation_diagnostics = {
                    "provider": "ollama",
                    "doneReason": result.get("done_reason"),
                    "promptTokenCount": result.get("prompt_eval_count"),
                    "generatedTokenCount": result.get("eval_count"),
                    "configuredOutputTokenLimit": (
                        cfg.capabilities.max_output_tokens
                    ),
                    "generatedOutputLimitReached": (
                        result.get("done_reason") == "length"
                    ),
                    "diagnosticMetadataFromProvider": True,
                    "semanticAccuracyVerified": False,
                }
                if result.get("done_reason") == "length":
                    raise LLMExecutionError("MODEL_GENERATION_TRUNCATED")
            else:
                choice = result["choices"][0]
                self._last_generation_diagnostics = {
                    "provider": "openai-compatible",
                    "finishReason": choice.get("finish_reason"),
                    "configuredOutputTokenLimit": (
                        cfg.capabilities.max_output_tokens
                    ),
                    "generatedOutputLimitReached": (
                        choice.get("finish_reason") == "length"
                    ),
                    "diagnosticMetadataFromProvider": True,
                    "semanticAccuracyVerified": False,
                }
                if choice.get("finish_reason") == "length":
                    raise LLMExecutionError("MODEL_GENERATION_TRUNCATED")
            if cfg.kind == "ollama":
                content = result["message"]["content"]
            else:
                content = result["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise LLMExecutionError("MODEL_EMPTY_CONTENT")
            if len(content.encode("utf-8")) > 30000:
                raise LLMExecutionError("MODEL_CONTENT_OVERSIZED")
            # Require valid JSON before downstream Pydantic + literal-source checks.
            json.loads(content)
            return content
        except LLMExecutionError:
            raise
        except httpx.TimeoutException as exc:
            raise LLMExecutionError("MODEL_TIMEOUT") from exc
        except httpx.HTTPError as exc:
            raise LLMExecutionError("MODEL_HTTP_OR_TRANSPORT_ERROR") from exc
        except (ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            raise LLMExecutionError("MODEL_INVALID_JSON_OR_RESPONSE") from exc
        finally:
            if owned:
                session.close()
