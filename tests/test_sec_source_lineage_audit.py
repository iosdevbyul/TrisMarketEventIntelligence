"""No retrospective SEC original overlap or self-reported witness clock promotion."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tris_market_event_intelligence.sec_source_lineage_audit_cli import (
    reconcile_source_lineage,
)

ACC1 = "0001045810-18-000004"
ACC2 = "0001193125-19-000003"
INV_SHA = "a" * 64
STUDY_SHA = "b" * 64
COHORT_SHA = "c" * 64
BATCH_SHA = "d" * 64


def _url(acc: str) -> str:
    return ("https://www.sec.gov/Archives/edgar/data/1045810/"
            + acc.replace("-", "") + "/" + acc + "-index.htm")


def _index(acc: str, form: str, day: str) -> bytes:
    return (f"<html><body>NVIDIA CORP Form {form} Accession Number "
            f"{acc} Filing Date {day} Accepted {day} 16:21:23"
            "</body></html>").encode()


def _inputs(directory: Path) -> tuple[dict, dict, dict]:
    events = [(ACC1, "8-K", "2018-02-08"), (ACC2, "10-Q", "2019-03-18")]
    for acc, form, day in events:
        (directory / f"sec-{acc}-index.html").write_bytes(_index(acc, form, day))
    rawsha = {
        acc: hashlib.sha256(_index(acc, form, day)).hexdigest()
        for acc, form, day in events
    }
    cohort = {
        "schemaVersion": "sec_pit_availability_research_cohort_v1",
        "status": "SEC_PUBLIC_AVAILABILITY_NOT_INDEPENDENTLY_OBSERVED",
        "ticker": "NVDA", "benchmark": "SPY",
        "selectionRule": "CALENDAR_YEAR_FORM_FIRST_MIDDLE_LAST_NO_PRICE_RANKING",
        "priceOutcomesUsedForSelection": False,
        "secReceiptClockIsNotPublicAvailability": True,
        "trainingReady": False, "predictionReady": False,
        "modelPromotionAllowed": False,
        "knownFirstPublicAvailabilityCases": 0,
        "multiyearStudySha256": STUDY_SHA,
        "reviewQueueCount": len(events),
        "filingCandidates": len(events),
        "reviewQueue": [
            {
                "accession": acc, "form": form, "filingDate": day,
                "secIndexUrl": _url(acc), "selectedWithoutFutureReturns": True,
                "alreadyRawPinned": acc == ACC1,
            } for acc, form, day in events
        ],
        "filingAvailabilityLedger": [
            {
                "accession": acc, "form": form, "filingDate": day,
                "secIndexUrl": _url(acc),
                "secIndexSha256": rawsha[acc] if acc == ACC1 else None,
                "publicAvailabilityVerifiedByCutoff": False,
                "usableAsPreopenModelFeature": False,
                "canBeUsedAsNoNewsControl": False,
            } for acc, form, day in events
        ],
    }
    study = {
        "schemaVersion": "historical_sec_multiyear_price_study_v1",
        "status": "RAW_SEC_SOURCES_PINNED_RETROSPECTIVE_PRICE_OUTCOMES",
        "ticker": "NVDA", "trainingReady": False, "eventCount": 1,
        "events": [{
            "accession": ACC1, "secIndexSha256": rawsha[ACC1],
            "issuerFilingIndexContentVerified": True,
            "trainingEligible": False, "outcome": {
                "horizonsTradingSessions": {"1": {"excessReturn": 0.5}}
            },
        }],
    }
    batch = {
        "schemaVersion": "sec_pit_original_index_capture_v1",
        "status": "SEC_ORIGINAL_INDEX_SHA_PINNED_PUBLICATION_TIME_UNPROVEN",
        "cohortSha256": COHORT_SHA,
        "ticker": "NVDA",
        "trainingReady": False, "backtestReady": False,
        "modelPromotionAllowed": False,
        "knownFirstPublicAvailabilityCases": 0,
        "secAcceptanceIsNotFirstPublicRelease": True,
        "secPublicDisseminationLagUnboundedHistorically": True,
        "selectedTaskCount": 2, "capturedIndexCount": 2,
        "previouslyCachedCount": 1, "newOriginalIndexCount": 1,
        "sourceRecords": [
            {
                "accession": acc, "form": form, "filingDate": day,
                "secIndexUrl": _url(acc), "secIndexSha256": rawsha[acc],
                "secAcceptedLocalUnverifiedTimezone": f"{day}T16:21:23",
                "historicalFirstPublicAvailabilityAt": None,
                "firstIndependentHistoricalObservedAt": None,
                "hasIndependentHistoricalPublicationClockEvidence": False,
                "verifiedPreopenMarketAvailability": False,
                "eligibleForNewsModelTraining": False,
                "secReceiptClockOnly": True,
                "secIndexMetadataCrossCheckedAgainstCohort": True,
                "fetchedInThisRun": acc == ACC2,
            } for acc, form, day in events
        ],
    }
    return cohort, study, batch


def _run(dir: Path, cohort: dict, study: dict, batch: dict,
         witnesses: dict | None = None) -> dict:
    return reconcile_source_lineage(
        cohort=cohort, cohort_sha=COHORT_SHA,
        study=study, study_sha=STUDY_SHA,
        batch=batch, batch_sha=BATCH_SHA,
        source_dir=dir, witnesses=witnesses,
        witness_sha="e" * 64 if witnesses is not None else None,
    )


def test_real_raw_bytes_overlap_dedup_and_no_outcome_leak(tmp_path: Path) -> None:
    cohort, study, batch = _inputs(tmp_path)
    result = _run(tmp_path, cohort, study, batch)
    assert result["capturedBatchOverlapWithOriginalTen"] == 1
    assert result["capturedBatchDistinctFromOriginalTen"] == 1
    assert result["uniqueSecIndexCasesAcrossBoth"] == 2
    assert result["overlappingAccessions"] == [ACC1]
    assert result["independentlyVerifiedFirstPublicTimeCount"] == 0
    assert result["trainingReady"] is False
    assert result["priceOutcomesReadForSelection"] is False
    assert "0.5" not in json.dumps(result)
    assert all(not case["predecisionEligibilityEstablished"]
               for case in result["cases"])


def test_cross_file_lineage_and_mismatched_sha_fail(tmp_path: Path) -> None:
    cohort, study, batch = _inputs(tmp_path)
    changed = copy.deepcopy(batch)
    changed["sourceRecords"][0]["secIndexSha256"] = "f" * 64
    with pytest.raises(ValueError, match="SEC_COHORT_AND_BATCH_SHA_CONFLICT"):
        _run(tmp_path, cohort, study, changed)
    changed = copy.deepcopy(cohort)
    changed["multiyearStudySha256"] = "f" * 64
    with pytest.raises(ValueError, match="SEC_LINEAGE_INPUT_PROVENANCE_INVALID"):
        _run(tmp_path, changed, study, batch)
    changed = copy.deepcopy(batch)
    changed["sourceRecords"][1]["historicalFirstPublicAvailabilityAt"] = (
        "2019-03-18T10:01:00-04:00"
    )
    with pytest.raises(ValueError, match="SEC_CAPTURE_RECORD_PROVENANCE_INVALID"):
        _run(tmp_path, cohort, study, changed)


def test_rehash_catches_mutation_without_redownload(tmp_path: Path) -> None:
    cohort, study, batch = _inputs(tmp_path)
    source = tmp_path / f"sec-{ACC2}-index.html"
    source.write_bytes(source.read_bytes().replace(b"NVIDIA", b"ANONXX"))
    with pytest.raises(ValueError, match="SEC_SOURCE_RAW_SHA_MISMATCH"):
        _run(tmp_path, cohort, study, batch)


def test_independent_source_bytes_are_not_timestamp_attestation(tmp_path: Path) -> None:
    cohort, study, batch = _inputs(tmp_path)
    witness_text = b"archived media item about NVIDIA"
    (tmp_path / "archived-news.html").write_bytes(witness_text)
    manifest = {
        "schemaVersion": "sec_historical_first_observation_witnesses_v1",
        "trainingReady": False, "independentAttestationComplete": False,
        "records": [{
            "witnessId": "archive_001", "accession": ACC1,
            "sourceType": "INDEPENDENT_MEDIA_ARCHIVE",
            "sourceFilename": "archived-news.html",
            "sourceUrl": "https://example.org/archived/nvidia",
            "sourceSha256": hashlib.sha256(witness_text).hexdigest(),
            "firstObservedAt": "2018-02-08T08:10:00-05:00",
            "publisherClaimedPublishedAt": "2018-02-08T08:00:00-05:00",
        }],
    }
    report = _run(tmp_path, cohort, study, batch, manifest)
    assert report["witnessClaimCount"] == 1
    assert report["independentlyVerifiedFirstPublicTimeCount"] == 0
    assert report["cases"][0]["historicalWitnessClaims"][0][
        "usableAsPredecisionFeature"
    ] is False
    changed = copy.deepcopy(manifest)
    changed["records"][0]["firstObservedAt"] = "2018-02-08T08:10:00"
    with pytest.raises(ValueError, match="SEC_WITNESS_TIMESTAMP_TIMEZONE_REQUIRED"):
        _run(tmp_path, cohort, study, batch, changed)
    changed = copy.deepcopy(manifest)
    changed["records"][0]["sourceSha256"] = "b" * 64
    with pytest.raises(ValueError, match="SEC_WITNESS_ORIGINAL_SHA_MISMATCH"):
        _run(tmp_path, cohort, study, batch, changed)
    changed = copy.deepcopy(manifest)
    changed["independentAttestationComplete"] = True
    with pytest.raises(ValueError, match="SEC_WITNESS_MANIFEST_INVALID"):
        _run(tmp_path, cohort, study, batch, changed)
