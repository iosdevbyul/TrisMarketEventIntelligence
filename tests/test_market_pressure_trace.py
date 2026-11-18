"""Verify real stage-level diagnostics survive both valid and invalid model calls."""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pytest
from test_market_evidence_db_bridge import populated_db
from test_market_event_discovery import InvalidPressureProvider

from tris_market_event_intelligence.market_event_discovery import discover_and_analyze
from tris_market_event_intelligence.market_pressure_trace import (
    MarketPressureTrace,
    TracedMarketProvider,
)


def _records(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_trace_durably_logs_real_stage_requests_and_invalid_responses(
    tmp_path: Path,
) -> None:
    db = populated_db(tmp_path / "news.sqlite3")
    path = tmp_path / "private-model-debug.jsonl"
    trace = MarketPressureTrace(path)
    try:
        provider = TracedMarketProvider(InvalidPressureProvider(), trace)
        result = discover_and_analyze(
            db, as_of=date(2024, 2, 15), provider=provider,
        )
        assert result["status"] == "MODEL_OUTPUT_REJECTED_REQUIRES_REVIEW"
        assert provider.calls == 3
    finally:
        trace.close()

    records = _records(path)
    requests = [entry for entry in records if entry["event"] == "LLM_REQUEST"]
    responses = [entry for entry in records if entry["event"] == "LLM_RESPONSE"]
    assert [x["stage"] for x in requests] == [
        "EVENT_GROUPING", "FACT_EXTRACTION", "IMPACT_GENERATION",
    ]
    assert [x["stage"] for x in responses] == [
        "EVENT_GROUPING", "FACT_EXTRACTION", "IMPACT_GENERATION",
    ]
    assert requests[1]["requestedSchema"]
    assert "increased input costs" in responses[1]["rawResponse"]
    assert "Semiconductor Market" in responses[2]["rawResponse"]
    assert any(x["event"] == "FACTS_VALIDATED" for x in records)
    errors = [x for x in records if x["event"] == "VALIDATION_REJECTED"]
    assert len(errors) == 1
    assert errors[0]["stage"] == "IMPACT_GENERATION"
    assert any(
        error["path"] == "impacts.0.claimIndexes"
        for error in errors[0]["fieldErrors"]
    )
    assert records[-1]["event"] == "TRACE_CLOSED"
    if os.name == "posix":
        assert path.stat().st_mode & 0o777 == 0o600


def test_trace_does_not_overwrite_prior_failed_run(tmp_path: Path) -> None:
    trace_file = tmp_path / "previous-diagnostic.jsonl"
    trace = MarketPressureTrace(trace_file)
    trace.close()
    before = trace_file.read_bytes()
    with pytest.raises(FileExistsError):
        MarketPressureTrace(trace_file)
    assert trace_file.read_bytes() == before


def test_trace_keeps_provider_failure_without_a_success_response(
    tmp_path: Path,
) -> None:
    class Failing:
        model_id = "fake-error"

        def generate_json(
            self, *, system: str, user: str, schema: dict | None,
        ) -> str:
            raise RuntimeError("MODEL_SIMULATED_FAILURE")

    path = tmp_path / "failing.jsonl"
    trace = MarketPressureTrace(path)
    try:
        provider = TracedMarketProvider(Failing(), trace)
        with pytest.raises(RuntimeError, match="MODEL_SIMULATED_FAILURE"):
            provider.generate_json(
                system="Identify related economic EVENT CLUSTERS",
                user='{"sourceIds":[]}', schema=None,
            )
    finally:
        trace.close()
    records = _records(path)
    assert any(x["event"] == "LLM_REQUEST" for x in records)
    assert any(x["event"] == "LLM_EXCEPTION" for x in records)
    assert not any(x["event"] == "LLM_RESPONSE" for x in records)
