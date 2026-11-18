"""Read-only automatic event grouping over collected news and indicators."""
from __future__ import annotations

import json
import sqlite3
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
            "targetName": "US equities", "strength": "UNASSESSED",
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


class InvalidPressureProvider(GroupAndImpactProvider):
    """Simulate the observed free-form multi-field Gemma4 output."""
    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        result = super().generate_json(system=system, user=user, schema=schema)
        if len(self.inputs) != 3:
            return result
        proposed = json.loads(result)
        impact = proposed["impacts"][0]
        impact["market"] = "Semiconductor Market"
        impact["sector"] = "Advanced Chip Manufacturing"
        impact["horizon"] = "Medium to Long-Term"
        impact["claimIndexes"] = []
        return json.dumps(proposed)


def test_malformed_gemma_impact_is_quarantined_without_inventing_signals(
    tmp_path: Path,
) -> None:
    db = populated_db(tmp_path / "news.sqlite3")
    provider = InvalidPressureProvider()
    result = discover_and_analyze(
        db, as_of=date(2024, 2, 15), provider=provider,
    )
    assert result["status"] == "MODEL_OUTPUT_REJECTED_REQUIRES_REVIEW"
    assert result["modelCalls"] == 3
    assert len(provider.inputs) == 3
    assert len(result["events"]) == 1
    rejected = result["events"][0]["analysis"]
    assert rejected["modelCalls"] == 2
    assert rejected["status"] == "MODEL_OUTPUT_REJECTED_REQUIRES_REVIEW"
    assert rejected["impacts"] == []
    assert rejected["validationStage"] == "IMPACT_GENERATION"
    assert len(rejected["groundedClaims"]) == 1
    assert rejected["groundedClaims"][0]["excerpt"] == "increased input costs"
    assert rejected["validationCode"] == "MODEL_SCHEMA_INVALID"
    fields = {item["path"] for item in rejected["invalidFields"]}
    assert "impacts.0.market" in fields
    assert "impacts.0.claimIndexes" in fields
    assert "Semiconductor Market" not in json.dumps(rejected)
    assert rejected["semanticAccuracyVerified"] is False
    assert rejected["marketReturnsAnalyzed"] is False


class ConflictingAbstentionProvider(GroupAndImpactProvider):
    """Reproduce one root-level Pydantic error, not a missing-source issue."""

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        result = super().generate_json(system=system, user=user, schema=schema)
        if len(self.inputs) == 3:
            record = json.loads(result)
            record["noImpactCategory"] = "OTHER"
            record["noImpactExplanation"] = (
                "The model declares that the event has no economic impact "
                "while also proposing a conditional market effect."
            )
            return json.dumps(record)
        return result


def test_root_validation_reason_and_first_pass_claims_are_not_lost(
    tmp_path: Path,
) -> None:
    db = populated_db(tmp_path / "news.sqlite3")
    provider = ConflictingAbstentionProvider()
    result = discover_and_analyze(
        db, as_of=date(2024, 2, 15), provider=provider,
    )
    analysis = result["events"][0]["analysis"]
    assert result["modelCalls"] == 3
    assert analysis["status"] == "MODEL_OUTPUT_REJECTED_REQUIRES_REVIEW"
    assert analysis["validationCode"] == "MODEL_SCHEMA_INVALID"
    assert analysis["validationReasonCode"] == "IMPACT_ABSTENTION_CONTRADICTION"
    assert analysis["invalidFields"] == [{"path": "", "type": "value_error"}]
    assert analysis["validationStage"] == "IMPACT_GENERATION"
    assert analysis["groundedClaims"][0]["sourceId"] == "article-1"
    assert analysis["impacts"] == []


class GeneralCausalDiscoveryProvider:
    """Exercise the actual discovery entry point with the new causal mode."""

    model_id = "mock-general-causal-discovery"

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.causal_schema_seen = False

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        supplied = json.loads(user)
        self.calls.append(supplied)
        if "cutoffVisibleEvidence" in supplied:
            return json.dumps({"groups": [{
                "sourceIndexes": [0, 1], "anchorIndex": 0,
                "connectionExplanation": (
                    "Two cutoff-visible observations potentially describe "
                    "conditional input costs and industrial demand."
                ),
            }]})
        if "visibleSources" in supplied:
            first = supplied["visibleSources"][0]
            quote = first["excerpt"][:min(150, len(first["excerpt"]))]
            return json.dumps({"claims": [{
                "sourceId": first["sourceId"],
                "excerpt": quote, "factKind": "REPORTED_FACT",
                "novelty": "UNKNOWN", "entities": [],
            }]})
        assert "groundedClaims" in supplied
        assert set(schema["properties"]) == {"candidateImpacts"}
        self.causal_schema_seen = True
        quote = supplied["groundedClaims"][0]["excerpt"]
        return json.dumps({"candidateImpacts": [{
            "targetType": "SECTOR",
            "targetName": "Industrial manufacturers",
            "market": "GLOBAL", "sector": "INDUSTRIALS",
            "horizon": "SHORT_TERM", "strength": "UNASSESSED",
            "causalPaths": [{
                "pressureDirection": "NEGATIVE",
                "evidenceClaimIndexes": [0],
                "observedExcerpt": quote,
                "economicSteps": [{
                    "cause": "The reported observation changes input costs",
                    "channel": "COST",
                    "effect": "Industrial margins may face economic pressure",
                    "condition": (
                        "Only if suppliers transmit those costs to buyers"
                    ),
                }],
                "competingExplanation": (
                    "Industry demand could offset this hypothetical pressure"
                ),
            }],
            "missingEvidence": ["No independent sector exposure figures"],
            "uncertaintyExplanation": (
                "This is a source-reported observation, not an independently "
                "measured change in sector earnings."
            ),
        }]})


def test_auto_discovery_routes_to_general_causal_engine(
    tmp_path: Path,
) -> None:
    db = populated_db(tmp_path / "news.sqlite3")
    provider = GeneralCausalDiscoveryProvider()
    report = discover_and_analyze(
        db, as_of=date(2024, 2, 15),
        provider=provider, reasoning_mode="causal",
    )
    assert provider.causal_schema_seen is True
    assert len(provider.calls) == 3
    assert report["modelCalls"] == 3
    assert report["status"] == "CANDIDATE_EVENTS_REQUIRE_REVIEW"
    assert len(report["events"]) == 1
    analysis = report["events"][0]["analysis"]
    assert analysis["analysisMode"] == "GENERAL_CAUSAL_PATHS_V1"
    assert analysis["modelCalls"] == 2
    assert len(analysis["impacts"]) == 1
    assert analysis["impacts"][0]["direction"] == "NEGATIVE"
    assert analysis["impacts"][0]["strength"] == "UNASSESSED"
    assert analysis["impacts"][0]["economicCausalityVerified"] is False
    assert analysis["semanticAccuracyVerified"] is False


def test_invalid_reasoning_mode_fails_before_model(tmp_path: Path) -> None:
    db = populated_db(tmp_path / "news.sqlite3")
    provider = GeneralCausalDiscoveryProvider()
    with pytest.raises(ValueError, match="INVALID_MARKET_REASONING_MODE"):
        discover_and_analyze(
            db, as_of=date(2024, 2, 15),
            provider=provider, reasoning_mode="per-company-rules",
        )
    assert provider.calls == []


def test_global_pagination_has_no_source_overlap_or_omissions(
    tmp_path: Path,
) -> None:
    db = populated_db(tmp_path / "news.sqlite3")
    with sqlite3.connect(db) as conn:
        for index in range(10, 67):
            day = 14 - (index - 10) // 15
            hour = (index - 10) % 15
            conn.execute(
                "INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?)",
                (
                    index, "feed-" + str(index % 3),
                    "https://example.org/unique/" + str(index),
                    "Distinct economic release " + str(index),
                    "An unrelated event about industry input " + str(index),
                    f"2024-02-{day:02d}T{hour:02d}:00:00+00:00",
                ),
            )
        for index in range(10, 23):
            conn.execute(
                "INSERT INTO economic_events VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    index, "POLICY",
                    "A separate policy event " + str(index)
                    + " changed an economic obligation.",
                    "policy-" + str(index),
                    "https://example.org/policy/" + str(index),
                    f"2024-02-13T{index-10:02d}:30:00+00:00",
                    "CANDIDATE",
                ),
            )
    pages = []
    inventory = []
    for page in range(5):
        candidates, info = recent_candidates(
            db, as_of=date(2024, 2, 15), lookback_days=14, page=page,
        )
        pages.append([row["evidence"].sourceId for row in candidates])
        inventory.append(info)
        if not info["hasMoreCandidatePages"]:
            break
    flattened = [sid for page in pages for sid in page]
    assert len(flattened) == len(set(flattened))
    assert len(flattened) == 72
    assert [len(page) for page in pages] == [24, 24, 24]
    assert [info["hasMoreCandidatePages"] for info in inventory] == [
        True, True, False,
    ]
    assert "economic-event-7" in flattened
    assert "article-1" in flattened
    balanced_pages = []
    for page in range(5):
        candidates, info = recent_candidates(
            db, as_of=date(2024, 2, 15), lookback_days=14,
            page=page, sampling_mode="source_balanced",
        )
        balanced_pages.append([row["evidence"].sourceId for row in candidates])
        assert info["samplingMode"] == "source_balanced"
        if not info["hasMoreCandidatePages"]:
            break
    balanced = [sid for page in balanced_pages for sid in page]
    assert len(balanced) == len(flattened)
    assert set(balanced) == set(flattened)
    assert len(balanced) == len(set(balanced))
    assert len([sid for sid in balanced_pages[0] if sid.startswith(
        "economic-event-"
    )]) >= 10
    assert len([sid for sid in pages[0] if sid.startswith(
        "economic-event-"
    )]) < 10
