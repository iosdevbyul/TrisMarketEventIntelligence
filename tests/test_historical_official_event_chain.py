"""Historical official FAA chain tests: actual independent dated sources."""
from __future__ import annotations

import copy
import json
import sys
from datetime import date
from pathlib import Path

import pytest

from tris_market_event_intelligence import historical_official_event_chain_cli as chain

COHORT = Path("research/faa-7379max-january-2024-event-chain.json")


def load() -> dict:
    return json.loads(COHORT.read_text(encoding="utf-8"))


class ChainProvider:
    model_id = "mock-chain"

    def __init__(self, *, inject_future: bool = False,
                 inject_wrong_edge: bool = False,
                 inject_market: bool = False):
        self.calls = 0
        self.inject_future = inject_future
        self.inject_wrong_edge = inject_wrong_edge
        self.inject_market = inject_market

    def generate_json(self, *, system: str, user: str, schema: dict) -> str:
        self.calls += 1
        packet = json.loads(user)
        assert packet["pointInTimeVerified"] is False
        assert packet["historicalPageSnapshotVerified"] is False
        assert packet["pricesAndOutcomesAvailable"] is False
        assert "excludedSources" not in packet
        assert "curatedEdges" not in packet
        assert "factInterpretations" in schema["properties"]
        references = [
            {"sourceId": source["sourceId"], "claimId": fact["claimId"],
             "interpretation": (
                 "The official agency published a dated action, not proof of "
                 "a share price impact or realized manufacturer expenses."
             ),
             "limitation": (
                 "This date-only source is not a verified intraday historical snapshot."
             )}
            for source in packet["eligibleSources"] for fact in source["claims"]
        ]
        if self.inject_future:
            references.append({
                "sourceId": "faa-20240124-production",
                "claimId": "no-production-expansion",
                "interpretation": (
                    "The agency later barred manufacturing capacity expansion "
                    "for the aircraft maker, which was not contemporaneously available."
                ),
                "limitation": (
                    "This cannot be grounded using sources from the earlier cutoff."
                ),
            })
        if self.inject_market and references:
            references[0]["interpretation"] = (
                "The stock shares fell by 5% following the announcement."
            )
        edges = [
            {"edgeId": edge["edgeId"],
             "fromSourceId": edge["fromSourceId"],
             "toSourceId": edge["toSourceId"],
             "interpretation": (
                 "This follow-up describes a later official procedural action "
                 "without proving the downstream manufacturer paid costs."
             ),
             "missingEvidence": (
                 "Independent operating and financial records would be needed."
             )}
            for edge in packet["visibleCuratedEdges"]
        ]
        if self.inject_wrong_edge and edges:
            edges[0]["toSourceId"] = "faa-20240106-grounding"
        return json.dumps({
            "factInterpretations": references,
            "edgeExplanations": edges,
            "whatChanged": (
                "The agency published several increasingly specific oversight "
                "and procedural announcements over this time period."
            ),
            "whatRemainsUnknown": (
                "The direct economic incidence of operational costs on Boeing "
                "is not established by this agency evidence alone."
            ),
        })


def test_real_faa_independent_source_dates_and_editorial_edges() -> None:
    doc = load()
    chain.validate_chain(doc)
    assert len(doc["sources"]) == 5
    assert len({s["url"] for s in doc["sources"]}) == 5
    assert len(doc["curatedEdges"]) == 3
    assert sum(len(s["claims"]) for s in doc["sources"]) == 10
    assert doc["allPublisherClocksVerified"] is False
    assert doc["historicalPageVersionsVerified"] is False


def test_early_replay_excludes_future_and_retrospective_sources() -> None:
    doc = load()
    packet = chain.prepare_replay(doc, date(2024, 1, 13))
    assert {s["sourceId"] for s in packet["eligibleSources"]} == {
        "faa-20240112-oversight", "faa-20240112-inspection-process",
    }
    assert packet["visibleCuratedEdges"] == []
    assert len(packet["excludedSources"]) == 3
    assert packet["pointInTimeVerified"] is False


def test_late_replay_follows_multisource_update_and_model_scope() -> None:
    doc = load()
    packet = chain.prepare_replay(doc, date(2024, 1, 25))
    assert len(packet["eligibleSources"]) == 4
    assert len(packet["visibleCuratedEdges"]) == 3
    assert "faa-20240106-grounding" not in {
        s["sourceId"] for s in packet["eligibleSources"]
    }
    assert {e["relation"] for e in packet["visibleCuratedEdges"]} == {
        "OFFICIAL_PROCESS_FOLLOWUP", "OVERSIGHT_ESCALATION",
        "RELATED_SCOPE_CHANGE",
    }
    assert packet["pointInTimeVerified"] is False


def test_same_day_release_not_replayed_without_verified_publication_clock() -> None:
    doc = load()
    jan12 = chain.prepare_replay(doc, date(2024, 1, 12))
    assert not jan12["eligibleSources"]
    jan24 = chain.prepare_replay(doc, date(2024, 1, 24))
    assert all(x["sourceDate"] < "2024-01-24"
               for x in jan24["eligibleSources"])


def test_actual_source_chain_two_cutoffs_uses_two_injected_model_calls() -> None:
    doc = load()
    fake = ChainProvider()
    early = chain.run_replay(doc, date(2024, 1, 13), fake)
    late = chain.run_replay(doc, date(2024, 1, 25), fake)
    assert fake.calls == 2
    assert early["assessment"]["availableSourceClaimCount"] == 4
    assert early["assessment"]["acceptedFactInterpretations"]
    assert early["assessment"]["visibleEdgeCount"] == 0
    assert late["assessment"]["availableSourceClaimCount"] == 8
    assert late["assessment"]["visibleEdgeCoverage"] == 3
    assert not late["assessment"]["rejectedEdgeExplanations"]
    assert not late["assessment"]["economicCausalityVerified"]
    assert not late["assessment"]["historicalPITVerified"]


def test_future_source_and_wrong_edges_are_quarantined() -> None:
    doc = load()
    result = chain.run_replay(
        doc, date(2024, 1, 13), ChainProvider(inject_future=True),
    )
    assert result["assessment"]["rejectedFactInterpretations"][0]["reason"] == (
        "FUTURE_OR_RETROSPECTIVE_SOURCE_REFERENCE"
    )
    later = chain.run_replay(
        doc, date(2024, 1, 25), ChainProvider(inject_wrong_edge=True),
    )
    assert later["assessment"]["rejectedEdgeExplanations"][0]["reason"] == (
        "ENDPOINTS_DISAGREE_WITH_CURATED_GRAPH"
    )


def test_market_outcome_not_in_official_research() -> None:
    doc = load()
    result = chain.run_replay(
        doc, date(2024, 1, 13), ChainProvider(inject_market=True),
    )
    assert result["assessment"]["rejectedFactInterpretations"][0]["reason"] == (
        "UNSUPPORTED_MARKET_OUTCOME"
    )


def test_malformed_sources_and_chronology_cannot_be_forged() -> None:
    src = load()
    invalid = copy.deepcopy(src)
    invalid["sources"][0]["asOfEligibility"] = "DATE_ONLY_CONSERVATIVE"
    with pytest.raises(ValueError, match="EVENT_CHAIN_RETROSPECTIVE_INVALID"):
        chain.validate_chain(invalid)
    # A retrospectively updated timeline cannot be made date-eligible
    # by changing only the source eligibility label.
    invalid = copy.deepcopy(src)
    invalid["curatedEdges"][0]["fromSourceId"] = "faa-20240124-production"
    with pytest.raises(ValueError, match="EVENT_CHAIN_EDGE_INVALID"):
        chain.validate_chain(invalid)


def test_preview_and_occupied_output_have_no_model_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", [
        "historical-event-chain", "--chain", str(COHORT),
        "--as-of", "2024-01-25", "--preview",
    ])
    chain.main()
    response = json.loads(capsys.readouterr().out)
    assert response["modelCalls"] == 0
    assert response["pointInTimeVerified"] is False
    assert len(response["eligibleSourceIds"]) == 4
    existing = tmp_path / "existing.json"
    existing.write_text('{"keep":true}', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "historical-event-chain", "--chain", str(COHORT),
        "--as-of", "2024-01-25", "--run", "--output", str(existing),
    ])
    monkeypatch.setattr(
        chain, "build_runtime",
        lambda *a, **k: pytest.fail("must preflight before model call"),
    )
    with pytest.raises(ValueError, match="Expected new private output"):
        chain.main()
    assert existing.read_text(encoding="utf-8") == '{"keep":true}'
