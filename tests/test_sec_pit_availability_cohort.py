"""PIT SEC cohort: no regulator receipt may become trusted news availability."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tris_market_event_intelligence.sec_pit_availability_cohort_cli import (
    _read_study,
    build_sec_pit_cohort,
)

INV_SHA = hashlib.sha256(b"inventory").hexdigest()
STUDY_SHA = hashlib.sha256(b"study").hexdigest()
ACC = (
    "0001045810-18-000001",
    "0001045810-18-000002",
    "0001045810-18-000003",
    "0001045810-19-000001",
    "0001045810-19-000002",
    "0001045810-19-000003",
)


def _inputs() -> tuple[dict, dict]:
    days = [
        "2018-01-12", "2018-06-20", "2018-12-04",
        "2019-02-01", "2019-07-08", "2019-12-18",
    ]
    forms = ["8-K", "8-K", "10-Q", "8-K", "8-K", "10-K"]
    filings = [{
        "accession": accession, "form": form, "filingDate": day,
        "secIndexUrl": (
            "https://www.sec.gov/Archives/edgar/data/1045810/"
            + accession.replace("-", "") + "/" + accession + "-index.htm"
        ),
        "eligibleAsNewsModelFeature": False,
    } for accession, form, day in zip(ACC, forms, days, strict=True)]
    inventory = {
        "schemaVersion": "sec_historical_filing_inventory_v1",
        "status":
        "SEC_SUBMISSIONS_CANDIDATES_REQUIRE_INDEPENDENT_TIME_AND_EXHIBIT_VERIFICATION",
        "companyCik": "0001045810", "ticker": "NVDA",
        "secAcceptanceTimeIndependentlyVerified": False,
        "originalFilingExhibitsPinned": False,
        "firstPublisherPublicationTimeVerified": False,
        "trainingReady": False, "candidateFilings": len(filings),
        "sourcePages": 1,
        "sources": [{"name": "CIK0001045810.json", "sha256": "c" * 64}],
        "filings": filings,
    }
    study = {
        "schemaVersion": "historical_sec_multiyear_price_study_v1",
        "status": "RAW_SEC_SOURCES_PINNED_RETROSPECTIVE_PRICE_OUTCOMES",
        "ticker": "NVDA", "benchmark": "SPY",
        "inventorySha256": INV_SHA, "stockSha256": "a" * 64,
        "benchmarkSha256": "b" * 64,
        "selectionRule": "EX_ANTE_FIRST_EIGHT_K_PER_YEAR_FROM_SEC_COVERAGE_REVIEW_QUEUE",
        "trainingReady": False, "predictionReady": False,
        "causalPriceImpactEstablished": False,
        "newsAbsenceVerifiedForControls": False,
        "firstPublisherPublicationTimeVerified": False,
        "priceMovementBeforeDecisionExcluded": False,
        "retrospectiveOutcomeAlreadyKnown": True,
        "modelPromotionAllowed": False,
        "rawSourceCount": 2,
        "eventCount": 1,
        "events": [{
            "accession": ACC[0], "secIndexSha256": "d" * 64,
            "secExhibit99_1Sha256": "e" * 64,
            "issuerFilingIndexContentVerified": True,
            "issuerExhibitContentVerified": True,
            "trainingEligible": False,
            "sourceAvailableByDecisionCutoff": False,
            "independentPublicAvailabilityAtCutoffVerified": False,
            "publisherFirstReleaseTimeVerified": False,
            "secReceiptBeforeDecisionCutoff": True,
            "secReceiptTiming": "AFTER_CLOSE_NEXT_SESSION",
            "secAcceptedAt": "2018-01-12T21:10:06+00:00",
            "eventSession": "2018-01-16",
            "decisionAt": "2018-01-16T14:30:00+00:00",
            "outcome": {
                "includesMovementBeforeDecision": True,
                "returnWindow": "PREVIOUS_CLOSE_TO_TARGET_CLOSE",
                "horizonsTradingSessions": {
                    horizon: {"excessReturn": 0.25, "stockReturn": 0.3}
                    for horizon in ("1", "3", "5")
                },
            },
        }],
    }
    return inventory, study


def _build(inventory: dict | None = None, study: dict | None = None,
           max_per_year_form: int = 3) -> dict:
    a, b = _inputs()
    return build_sec_pit_cohort(
        inventory=inventory if inventory is not None else a,
        inventory_sha=INV_SHA, study=study if study is not None else b,
        study_sha=STUDY_SHA, max_per_year_form=max_per_year_form,
    )


def test_all_six_filings_tracked_without_outcome_leakage() -> None:
    result = _build()
    assert result["filingCandidates"] == 6
    assert result["secRawPinnedCases"] == 1
    assert result["metadataOnlyCases"] == 5
    assert result["knownFirstPublicAvailabilityCases"] == 0
    assert result["reviewQueueCount"] == 6
    assert result["priceOutcomesUsedForSelection"] is False
    assert result["trainingReady"] is False
    assert result["backtestReady"] is False
    assert len(result["filingAvailabilityLedger"]) == 6
    assert all(row["publicAvailabilityVerifiedByCutoff"] is False
               and row["marketAvailableAt"] is None
               and row["canBeUsedAsNoNewsControl"] is False
               for row in result["filingAvailabilityLedger"])
    assert all(row["selectedWithoutFutureReturns"] is True
               for row in result["reviewQueue"])
    assert "0.25" not in json.dumps(result)
    pinned = result["filingAvailabilityLedger"][0]
    assert pinned["secIndexSha256"] == "d" * 64
    assert pinned["secExhibit99_1Sha256"] == "e" * 64
    assert pinned["secReceiptAt"] == "2018-01-12T21:10:06+00:00"
    assert pinned["evidenceStatus"] == (
        "SEC_RAW_INDEX_PINNED_PUBLICATION_CLOCK_UNKNOWN"
    )


def test_bounded_stratified_selection_does_not_change_with_outcomes() -> None:
    original, study = _inputs()
    study_bad = copy.deepcopy(study)
    study_bad["events"][0]["outcome"]["horizonsTradingSessions"]["1"][
        "excessReturn"
    ] = -0.99
    a = _build(inventory=original, study=study, max_per_year_form=1)
    b = _build(inventory=original, study=study_bad, max_per_year_form=1)
    assert a["reviewQueue"] == b["reviewQueue"]
    assert a["reviewQueueCount"] == 4
    assert len(_build(max_per_year_form=2)["reviewQueue"]) == 6


def test_fabricated_public_availability_and_training_rejected() -> None:
    inv, study = _inputs()
    for field in ("sourceAvailableByDecisionCutoff",
                  "independentPublicAvailabilityAtCutoffVerified",
                  "trainingEligible"):
        modified = copy.deepcopy(study)
        modified["events"][0][field] = True
        with pytest.raises(ValueError, match="SEC_STUDY_SOURCE_ATTESTATION_INVALID"):
            _build(inventory=inv, study=modified)
    modified = copy.deepcopy(study)
    modified["trainingReady"] = True
    with pytest.raises(ValueError, match="SEC_STUDY_PIT_PROVENANCE_INVALID"):
        _build(inventory=inv, study=modified)


def test_issuer_and_source_lineage_mismatches_rejected() -> None:
    inv, study = _inputs()
    changed = copy.deepcopy(inv)
    changed["filings"][0]["accession"] = "0001045810-18-999999"
    with pytest.raises(ValueError, match="SEC_REVIEWED_EVENT_NOT_IN_INVENTORY"):
        _build(inventory=changed, study=study)
    changed = copy.deepcopy(inv)
    changed["filings"][0]["eligibleAsNewsModelFeature"] = True
    with pytest.raises(ValueError, match="SEC_INVENTORY_ROW_PROVENANCE_INVALID"):
        _build(inventory=changed, study=study)
    changed_study = copy.deepcopy(study)
    changed_study["eventCount"] = 11
    with pytest.raises(ValueError, match="SEC_STUDY_EVENT_COUNT_INVALID"):
        _build(inventory=inv, study=changed_study)
    changed_study = copy.deepcopy(study)
    changed_study["inventorySha256"] = "f" * 64
    with pytest.raises(ValueError, match="SEC_INVENTORY_SHA_NOT_STUDY_PINNED"):
        _build(inventory=inv, study=changed_study)


def test_read_study_private_bytes_sha_and_symlinks(tmp_path: Path) -> None:
    _, study = _inputs()
    study_path = tmp_path / "study.json"
    study_path.write_text(json.dumps(study))
    obj, digest = _read_study(study_path)
    assert obj["eventCount"] == 1
    assert len(digest) == 64
    link = tmp_path / "symlink.json"
    link.symlink_to(study_path)
    with pytest.raises(ValueError, match="SEC_STUDY_FILE_INVALID"):
        _read_study(link)
