"""Offline, zero-model-call diagnostics for v7 impact-stage guard failures."""
from __future__ import annotations

import json
import runpy
from pathlib import Path


def test_trace_reports_uncited_entities_and_uncertain_strength(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "market-private.jsonl"
    claims = [
        {"index": 0, "excerpt": "Pakistan grid solar share climbed to 28%"},
        {"index": 1, "excerpt": "Nvidia quarterly earnings improved"},
    ]
    impacts = [
        {
            "direction": "NEGATIVE",
            "strength": "LOW",
            "targetType": "SECTOR",
            "targetName": "Pakistan solar",
            "claimIndexes": [0],
            "entities": ["Pakistan", "Solar Supply Incorporated"],
            "evidence_excerpt": "Pakistan grid solar share",
        },
        {
            "direction": "UNCERTAIN",
            "strength": "LOW",
            "targetType": "COMPANY",
            "targetName": "Nvidia",
            "claimIndexes": [1],
            "entities": ["Nvidia"],
            "evidence_excerpt": "Nvidia quarterly earnings",
        },
    ]
    lines = [
        {"event": "LLM_REQUEST", "stage": "IMPACT_GENERATION",
         "call": 3, "userInput": json.dumps({"groundedClaims": claims})},
        {"event": "LLM_RESPONSE", "stage": "IMPACT_GENERATION",
         "call": 3, "rawResponse": json.dumps({"impacts": impacts})},
    ]
    trace.write_text(
        "\n".join(json.dumps(x) for x in lines), encoding="utf-8",
    )
    scope = runpy.run_path("scripts/diagnose-market-pressure-trace.py")
    result = scope["inspect_trace"](trace)
    assert result["modelCalls"] == 0
    assert result["readOnly"] is True
    assert result["impactCalls"][0]["impactCount"] == 2
    first, second = result["impactCalls"][0]["impacts"]
    assert first["uncitedEntities"] == ["Solar Supply Incorporated"]
    assert first["directionStrengthConflict"] is False
    assert first["evidenceExcerptMatchesCitedClaims"] is True
    assert second["uncitedEntities"] == []
    assert second["directionStrengthConflict"] is True
    assert second["validClaimIndexes"] is True
