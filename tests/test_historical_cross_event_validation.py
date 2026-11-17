"""Official FDA independent event history and FAA holdout comparison."""
from __future__ import annotations

import copy
import json
import sys
from datetime import date
from pathlib import Path

import pytest

from test_historical_official_event_chain import ChainProvider, load

from tris_market_event_intelligence import historical_cross_event_validation_cli as cross
from tris_market_event_intelligence import historical_official_event_chain_cli as chain

FDA = Path("research/fda-covid-vaccine-2023-event-chain.json")


def fda() -> dict:
    return json.loads(FDA.read_text(encoding="utf-8"))


def test_fda_four_verified_dated_release_sources_and_three_real_links() -> None:
    doc = fda()
    chain.validate_chain(doc)
    assert len(doc["sources"]) == 4
    assert len({source["url"] for source in doc["sources"]}) == 4
    assert [source["sourceDate"] for source in doc["sources"]] == [
        "2023-04-18", "2023-06-16", "2023-09-11", "2023-10-03",
    ]
    assert len(doc["curatedEdges"]) == 3
    assert sum(len(source["claims"]) for source in doc["sources"]) == 8
    assert doc["allPublisherClocksVerified"] is False


def test_heldout_fda_conservative_two_date_views() -> None:
    doc = fda()
    early = chain.prepare_replay(doc, date(2023, 6, 17))
    late = chain.prepare_replay(doc, date(2023, 10, 4))
    assert [x["sourceId"] for x in early["eligibleSources"]] == [
        "fda-20230418-bivalent", "fda-20230616-formula",
    ]
    assert len(early["visibleCuratedEdges"]) == 1
    assert len(late["eligibleSources"]) == 4
    assert len(late["visibleCuratedEdges"]) == 3
    assert len(chain.prepare_replay(doc, date(2023, 10, 3))["eligibleSources"]) == 3
    assert chain.compare_replays(early, late)["newlyEligibleSourceIds"] == [
        "fda-20230911-mrna", "fda-20231003-novavax",
    ]
    assert early["pointInTimeVerified"] is False


def test_fda_sequence_works_with_existing_schema_and_quality_engine() -> None:
    doc = fda()
    provider = ChainProvider()
    result = chain.run_sequence(
        doc, [date(2023, 6, 17), date(2023, 10, 4)], provider,
    )
    assert provider.calls == 2
    assert [x["assessment"]["availableSourceClaimCount"]
            for x in result["replaySteps"]] == [4, 8]
    assert [x["assessment"]["visibleEdgeCoverage"]
            for x in result["replaySteps"]] == [1, 3]
    assert result["historicalPITVerified"] is False
    assert result["semanticAccuracyVerified"] is False


def test_source_host_and_mixed_agency_cannot_be_spoofed() -> None:
    doc = fda()
    forged = copy.deepcopy(doc)
    forged["sources"][0]["url"] = "https://www.faa.gov/newsroom/example"
    with pytest.raises(ValueError, match="EVENT_CHAIN_SOURCE_INVALID"):
        chain.validate_chain(forged)
    forged = copy.deepcopy(doc)
    forged["sources"][0]["publisher"] = "FAA"
    forged["sources"][0]["url"] = "https://www.faa.gov/newsroom/example"
    with pytest.raises(ValueError, match="EVENT_CHAIN_MIXED_AGENCY_INVALID"):
        chain.validate_chain(forged)


def test_heldout_chain_comparison_distinguishes_factual_vs_semantic_scores() -> None:
    faa = load()
    fda_doc = fda()
    first = chain.run_sequence(
        faa, [date(2024, 1, 13), date(2024, 1, 25)], ChainProvider(),
    )
    second = chain.run_sequence(
        fda_doc, [date(2023, 6, 17), date(2023, 10, 4)], ChainProvider(),
    )
    old_first = copy.deepcopy(first)
    old_second = copy.deepcopy(second)
    audit = cross.compare_independent_events(faa, first, fda_doc, second)
    assert first == old_first and second == old_second
    assert audit["sourceIndependence"]["sourceIdOverlap"] == 0
    assert audit["sourceIndependence"]["sourceUrlOverlap"] == 0
    assert audit["fairLikeForLikeModelSchemaComparison"] is True
    assert audit["auditModelCalls"] == 0
    assert audit["evidenceOfCrossEventGeneralization"] is False
    assert audit["primary"]["steps"][1]["allReferencesCovered"] is True
    assert audit["independentHoldout"]["steps"][1]["allReferencesCovered"] is True
    assert audit["historicalPITVerified"] is False
    assert audit["promotionAllowed"] is False


def test_legacy_faa_v1_vs_holdout_v2_not_falsely_comparable() -> None:
    faa = load()
    fda_doc = fda()
    baseline = chain.run_sequence(
        faa, [date(2024, 1, 13), date(2024, 1, 25)], ChainProvider(),
    )
    baseline["schemaVersion"] = "historical_official_event_chain_sequence_v1"
    heldout = chain.run_sequence(
        fda_doc, [date(2023, 6, 17), date(2023, 10, 4)], ChainProvider(),
    )
    audit = cross.compare_independent_events(faa, baseline, fda_doc, heldout)
    assert audit["fairLikeForLikeModelSchemaComparison"] is False
    assert audit["evidenceOfCrossEventGeneralization"] is False


def test_saved_holdout_artifact_scope_forgery_rejected() -> None:
    faa = load()
    fda_doc = fda()
    first = chain.run_sequence(
        faa, [date(2024, 1, 13), date(2024, 1, 25)], ChainProvider(),
    )
    second = chain.run_sequence(
        fda_doc, [date(2023, 6, 17), date(2023, 10, 4)], ChainProvider(),
    )
    second["replaySteps"][0]["eligibleSources"].append(
        fda_doc["sources"][-1]
    )
    with pytest.raises(ValueError, match="EVENT_CHAIN_SAVED_SOURCE_SCOPE_INVALID"):
        cross.compare_independent_events(faa, first, fda_doc, second)


def test_offline_cli_no_model_calls_and_refuses_occupied_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    primary = load()
    heldout = fda()
    first = chain.run_sequence(
        primary, [date(2024, 1, 13), date(2024, 1, 25)], ChainProvider(),
    )
    second = chain.run_sequence(
        heldout, [date(2023, 6, 17), date(2023, 10, 4)], ChainProvider(),
    )
    p1, p2, o = (tmp_path / name for name in ["first.json", "second.json", "audit.json"])
    p1.write_text(json.dumps(first), encoding="utf-8")
    p2.write_text(json.dumps(second), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "cross-event",
        "--primary-chain", "research/faa-7379max-january-2024-event-chain.json",
        "--primary-existing", str(p1),
        "--holdout-chain", str(FDA),
        "--holdout-existing", str(p2),
        "--output", str(o),
    ])
    cross.main()
    output = json.loads(capsys.readouterr().out)
    assert output["auditModelCalls"] == 0
    assert o.exists()
    with pytest.raises(ValueError, match="Expected new private output"):
        cross.main()
