"""Candidates from SEC history must not become fake historic news-negative labels."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tris_market_event_intelligence.sec_coverage_reconciliation_cli import (
    read_inventory,
    reconcile_sec_coverage,
)


def _inventory() -> dict:
    filings = []
    for accession, day, form in [
        ("0001045810-18-000147", "2018-11-14", "8-K"),
        ("0001045810-18-000148", "2018-11-15", "8-K"),
        ("0001045810-18-000149", "2018-11-16", "10-Q"),
        ("0001045810-18-000150", "2018-11-20", "10-K"),
    ]:
        filings.append({
            "accession": accession,
            "form": form,
            "filingDate": day,
            "rawSecAcceptanceDateTime": f"{day}T17:00:00",
            "secIndexUrl": (
                "https://www.sec.gov/Archives/edgar/data/1045810/"
                f"{accession.replace('-', '')}/{accession}-index.htm"
            ),
            "secAcceptanceTimeIndependentlyVerified": False,
            "filingExhibitFactsVerified": False,
            "eligibleAsNewsModelFeature": False,
        })
    return {
        "schemaVersion": "sec_historical_filing_inventory_v1",
        "status": (
            "SEC_SUBMISSIONS_CANDIDATES_REQUIRE_INDEPENDENT_TIME_AND_EXHIBIT_VERIFICATION"
        ),
        "companyCik": "0001045810",
        "ticker": "NVDA",
        "secAcceptanceTimeIndependentlyVerified": False,
        "originalFilingExhibitsPinned": False,
        "firstPublisherPublicationTimeVerified": False,
        "trainingReady": False,
        "sourcePages": 1,
        "sources": [{"name": "CIK0001045810.json", "sha256": "a" * 64}],
        "candidateFilings": len(filings),
        "filings": filings,
    }


def _universe() -> dict:
    dates = ["2018-11-14", "2018-11-15", "2018-11-16", "2018-11-19"]
    records = []
    for day in dates:
        event = ([{"secAccession": "0001045810-18-000148"}]
                 if day == "2018-11-16" else [])
        records.append({
            "session": day,
            "features": {"pinnedSecEvents": event},
            "trainingEligible": False,
        })
    return {
        "schemaVersion": "historical_sec_event_universe_v1",
        "status": "MULTI_SESSION_RETROSPECTIVE_SEC_EVIDENCE",
        "ticker": "NVDA",
        "benchmark": "SPY",
        "stockSha256": "b" * 64,
        "benchmarkSha256": "c" * 64,
        "nonEventSessionsAreTrueNoNewsControls": False,
        "trainingReady": False,
        "evidenceCoverageComplete": False,
        "sessionCount": len(records),
        "eventCount": 1,
        "sessionsWithPinnedSecEvent": 1,
        "sessionsWithoutPinnedSecEvent": 3,
        "records": records,
    }


def _run(**kwargs: object) -> dict:
    return reconcile_sec_coverage(
        inventory=kwargs.get("inventory", _inventory()),
        inventory_sha="d" * 64,
        universe=kwargs.get("universe", _universe()),
        max_candidates=kwargs.get("max_candidates", 200),
    )


def test_official_sec_match_and_unverified_candidates_are_distinct() -> None:
    result = _run()
    assert result["pinnedAccessionMatchesInInventory"] == 1
    assert result["independentlyPinnedEventCount"] == 1
    assert result["unverifiedCandidateCount"] == 3
    assert result["inventoryFormCounts"] == {"10-K": 1, "10-Q": 1, "8-K": 2}
    assert result["unknownNewsStatusSessionCount"] == 3
    assert result["noCandidateDoesNotMeanNoNews"] is True
    assert result["trainingReady"] is False
    assert result["backtestReady"] is False
    assert result["reviewQueue"][0]["candidateSessions"] == [
        "2018-11-14", "2018-11-15",
    ]
    assert result["reviewQueue"][1]["candidateSessions"] == [
        "2018-11-16", "2018-11-19",
    ]
    assert result["reviewQueue"][2]["candidateSessions"] == []
    assert all(item["eligibleForPredictionFeatures"] is False
               for item in result["reviewQueue"])
    assert "excessReturn" not in json.dumps(result)
    assert result["verifiedWithinInventory"][0]["verifiedEventSession"] == (
        "2018-11-16"
    )


def test_truncated_queue_only_affects_display_not_provenance() -> None:
    result = _run(max_candidates=1)
    assert result["reviewQueueTruncated"] is True
    assert result["unverifiedCandidateCount"] == 3
    assert len(result["reviewQueue"]) == 1


def test_rejects_fabricated_verified_sec_clock() -> None:
    doc = _inventory()
    doc["filings"][0]["secAcceptanceTimeIndependentlyVerified"] = True
    with pytest.raises(ValueError, match="CANDIDATE_CONTRACT"):
        _run(inventory=doc)


def test_rejects_promoted_control_or_training_row() -> None:
    universe = _universe()
    universe["nonEventSessionsAreTrueNoNewsControls"] = True
    with pytest.raises(ValueError, match="PROVENANCE"):
        _run(universe=universe)
    universe = _universe()
    universe["records"][0]["trainingEligible"] = True
    with pytest.raises(ValueError, match="TRAINING_PROMOTION"):
        _run(universe=universe)


def test_rejects_duplicate_and_issuer_mismatch() -> None:
    doc = _inventory()
    doc["filings"].append(copy.deepcopy(doc["filings"][0]))
    doc["candidateFilings"] += 1
    with pytest.raises(ValueError, match="NOT_UNIQUE"):
        _run(inventory=doc)
    doc = _inventory()
    doc["ticker"] = "MSFT"
    with pytest.raises(ValueError, match="PROVENANCE"):
        _run(inventory=doc)


def test_read_inventory_is_bounded_and_has_file_sha(tmp_path: Path) -> None:
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(_inventory()))
    result, digest = read_inventory(path)
    assert len(result["filings"]) == 4
    assert len(digest) == 64
    link = tmp_path / "inventory-link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="FILE_INVALID"):
        read_inventory(link)
