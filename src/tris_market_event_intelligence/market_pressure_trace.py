"""Private, durable, opt-in diagnostics for actual market-pressure model runs.

Each JSONL record is flushed and fsynced independently so partial runs and
validation failures retain original prompts, replies, schema and error details.
This trace is PRIVATE LOCAL DATA: never publish it to GitHub or an API.
"""
from __future__ import annotations

import hashlib
import json
import os
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from tris_market_event_intelligence.llm_runtime import LLMProvider


def _stage(system: str) -> str:
    if "Identify related economic EVENT CLUSTERS" in system:
        return "EVENT_GROUPING"
    if "financial news and economic-indicator evidence reader" in system:
        return "FACT_EXTRACTION"
    if "CONDITIONAL economic MARKET PRESSURE" in system:
        return "IMPACT_GENERATION"
    return "UNKNOWN_MODEL_STAGE"


def _checksum(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class MarketPressureTrace:
    """Create-only 0600 JSONL file; do not silently overwrite earlier runs."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        self._file = os.fdopen(fd, "w", encoding="utf-8")
        self.count = 0
        self.record(
            "TRACE_OPENED",
            note="Sensitive local-only model prompts and responses: do not upload",
        )

    def record(self, event: str, **details: Any) -> None:
        self.count += 1
        row = {
            "sequence": self.count,
            "recordedAtUTC": datetime.now(UTC).isoformat(),
            "event": event,
            **details,
        }
        self._file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())

    def validation_failure(
        self, *, stage: str, error: BaseException,
        event_id: str | None = None,
    ) -> None:
        details: list[dict] = []
        if isinstance(error, ValidationError):
            details = [
                {
                    "path": ".".join(str(part) for part in issue["loc"]),
                    "type": issue["type"],
                    "message": issue["msg"],
                }
                for issue in error.errors(
                    include_input=False, include_context=False,
                )
            ]
        self.record(
            "VALIDATION_REJECTED", stage=stage, eventId=event_id,
            errorType=type(error).__name__,
            errorMessage=str(error),
            fieldErrors=details,
            traceback="".join(traceback.format_exception(error)),
        )

    def close(self) -> None:
        if not self._file.closed:
            self.record("TRACE_CLOSED")
            self._file.close()


class TracedMarketProvider:
    """Transparent decorator around one real LLM; never retries generation."""

    def __init__(self, provider: LLMProvider, trace: MarketPressureTrace) -> None:
        self._provider = provider
        self.trace = trace
        self.model_id = provider.model_id
        self.settings = getattr(provider, "settings", None)
        self.calls = 0

    def generate_json(
        self, *, system: str, user: str, schema: dict | None,
    ) -> str:
        self.calls += 1
        stage = _stage(system)
        call = self.calls
        self.trace.record(
            "LLM_REQUEST", stage=stage, call=call,
            modelId=self.model_id, systemPrompt=system,
            userInput=user, requestedSchema=schema,
            promptSha256=_checksum(system + "\n" + user),
        )
        try:
            result = self._provider.generate_json(
                system=system, user=user, schema=schema,
            )
        except Exception as error:
            self.trace.record(
                "LLM_EXCEPTION", stage=stage, call=call,
                errorType=type(error).__name__,
                errorMessage=str(error),
                traceback="".join(traceback.format_exception(error)),
            )
            raise
        self.trace.record(
            "LLM_RESPONSE", stage=stage, call=call,
            rawResponse=result, responseSha256=_checksum(result),
            responseChars=len(result),
        )
        return result
