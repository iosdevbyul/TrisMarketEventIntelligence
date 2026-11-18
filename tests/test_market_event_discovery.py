"""Read-only automatic event grouping over collected news and indicators."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from test_market_evidence_db_bridge import populated_db

from tris_market_event_intelligence.market_event_discovery import (
    discover_and_analyze,
    recent_candidates,
)


class GroupAndImpactProvider:
    model_id = "mock-auto-event"

    def __init__(self, invalid_index: bool = False) -> None:
        self.inputs: list[dict] = []
        self.invalid_index = invalid_index

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        self.inputs.append(json.loads(user))
        if len(self.inputs) == 1:
            return json.dumps({"groups": [{
                "sourceIndexes": [999 if self.invalid_index else 0, 1],
                "anchorIndex": 999 if self.invalid_index else 0,
                "connectionExplanation": (
                    "Both sources discuss economic conditions "
                    "and their possible effects on financing."
                ),
            }]})
        if len(self.inputs) == 2:
            return json.dumps({"claims": [{
                "sourceId": "article-1",
                "excerpt": "increased input costs",
                "factKind": "REPORTED_FACT",
                "novelty": "UNKNOWN",
                "entities": [],
            }]})
        return json.dumps({"impacts": [{
            "market": "US", "sector": "BROAD_MARKET", "direction": "NEGATIVE",
            "horizon": "SHORT_TERM",
            "rationale": (
                "The reported higher input costs could compress company "
                "earnings expectations and conditionally weaken equity "
                "valuations, but no consensus or priced-in information is known."
            ),
            "evidence_excerpt": "increased input costs",
            "uncertainty": "HIGH", "drivers": [],
            "entities": [], "targetType": "MARKET",
            "targetName": "US equities", "strength": "UNKNOWN",
            "strengthRationale": (
                "No sourced basis allows a quantified pressure magnitude."
            ),
            "transmissionChannel": "COST",
            "claimIndexes": [0],
            "countervailingFactors": [],
        }]})


def test_auto_cycle_identifies_and_analyzes_only_cutoff_evidence(
    tmp_path: Path,
) -> None:
    db = populated_db(tmp_path / "news.sqlite3")
    provider = GroupAndImpactProvider()
    result = discover_and_analyze(
        db, as_of=date(2024, 2, 15), provider=provider,
    )
    assert result["modelCalls"] == 3
    assert len(result["events"]) == 1
    assert result["events"][0]["groupingVerified"] is False
    analysis = result["events"][0]["analysis"]
    assert analysis["modelCalls"] == 2
    assert analysis["impacts"][0]["direction"] == "NEGATIVE"
    assert analysis["impacts"][0]["supportingSourceIds"] == ["article-1"]
    assert all("future" not in json.dumps(x).lower()
               for x in provider.inputs)
    assert result["marketReturnsAnalyzed"] is False


def test_invalid_group_indexes_rejected_before_followup_model_calls(
    tmp_path: Path,
) -> None:
    db = populated_db(tmp_path / "news.sqlite3")
    provider = GroupAndImpactProvider(invalid_index=True)
    with pytest.raises(ValueError, match="EVENT_GROUP_SOURCE_INDEX_OUT_OF_BOUNDS"):
        discover_and_analyze(
            db, as_of=date(2024, 2, 15), provider=provider,
        )
    assert len(provider.inputs) == 1


def test_no_dated_candidates_needs_no_model(tmp_path: Path) -> None:
    db = populated_db(tmp_path / "news.sqlite3")
    provider = GroupAndImpactProvider()
    result = discover_and_analyze(
        db, as_of=date(2024, 2, 1), provider=provider,
    )
    assert result["status"] == "NO_ELIGIBLE_DATED_SOURCES"
    assert result["modelCalls"] == 0
    assert provider.inputs == []


def test_recent_candidates_are_read_only_and_bounded(tmp_path: Path) -> None:
    db = populated_db(tmp_path / "news.sqlite3")
    candidates, inventory = recent_candidates(
        db, as_of=date(2024, 2, 15), lookback_days=7,
    )
    assert inventory["candidateCount"] == 2
    assert {row["evidence"].sourceId for row in candidates} == {
        "article-1", "economic-event-7",
    }
    assert len(candidates) <= 24
    assert all(row["evidence"].verification == "REPORTED_UNVERIFIED"
               for row in candidates)
