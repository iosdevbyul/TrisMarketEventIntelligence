"""Replay exact saved provider outputs with no model or network access."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_market_pressure_reasoning import example, first_pass, impact

from tris_market_event_intelligence.market_pressure_replay_cli import replay_trace


def _trace_fixture(path: Path, *, missing_impact: bool = False) -> None:
    packet = example()
    prompt = {
        "eventId": packet.eventId,
        "subject": packet.subject,
        "asOfDateExclusive": packet.asOfDateExclusive.isoformat(),
        "asOfTimestampExclusive": None,
        "visibleSources": [
            source.model_dump(mode="json")
            for source in packet.evidences[:2]
        ],
    }
    bad = impact()
    bad["entities"] = ["Pakistan national grid"]
    good = impact()
    lines = [
        {
            "event": "LLM_REQUEST", "stage": "FACT_EXTRACTION",
            "call": 2, "modelId": "recorded-gemma",
            "userInput": json.dumps(prompt),
        },
        {
            "event": "LLM_RESPONSE", "stage": "FACT_EXTRACTION",
            "call": 2, "rawResponse": json.dumps(first_pass()),
        },
        {
            "event": "LLM_REQUEST", "stage": "IMPACT_GENERATION",
            "call": 3, "modelId": "recorded-gemma",
            "userInput": json.dumps({
                "eventId": packet.eventId,
                "groundedClaims": [
                    {"index": index, **claim, "sourceVerification": "REPORTED_UNVERIFIED"}
                    for index, claim in enumerate(first_pass()["claims"])
                ],
            }),
        },
    ]
    if not missing_impact:
        lines.append({
            "event": "LLM_RESPONSE", "stage": "IMPACT_GENERATION",
            "call": 3, "rawResponse": json.dumps({
                "impacts": [bad, good],
            }),
        })
    path.write_text(
        "\n".join(json.dumps(row) for row in lines) + "\n",
        encoding="utf-8",
    )


def test_offline_replay_retains_valid_and_diagnoses_invalid(tmp_path: Path) -> None:
    path = tmp_path / "recorded-private.jsonl"
    _trace_fixture(path)
    before = path.read_bytes()
    result = replay_trace(path)
    assert path.read_bytes() == before
    assert result["newModelCalls"] == 0
    assert result["dbWrites"] == 0
    assert result["acceptedImpactCount"] == 1
    assert result["rejectedImpactCount"] == 1
    group = result["events"][0]
    assert group["status"] == "PARTIAL_MARKET_PRESSURE_REQUIRES_REVIEW"
    assert group["acceptedImpacts"][0]["proposalIndex"] == 1
    assert group["rejectedImpacts"][0]["code"] == "IMPACT_ENTITY_MISMATCH"
    assert group["semanticAccuracyVerified"] is False


def test_offline_replay_does_not_regenerate_missing_model_response(
    tmp_path: Path,
) -> None:
    path = tmp_path / "incomplete-private.jsonl"
    _trace_fixture(path, missing_impact=True)
    with pytest.raises(ValueError, match="REPLAY_MODEL_RESPONSE_MISSING"):
        replay_trace(path)
