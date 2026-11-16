"""SEC official index capture must never hallucinate historical public clocks."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import httpx
import pytest

from tris_market_event_intelligence.sec_publication_evidence_capture_cli import (
    _url_allowed,
    _validate_cohort,
    capture_sec_pit_evidence,
)

ACC1 = "0001045810-18-000004"
ACC2 = "0001193125-19-000003"


def _url(acc: str) -> str:
    return ("https://www.sec.gov/Archives/edgar/data/1045810/"
            + acc.replace("-", "") + "/" + acc + "-index.htm")


def _index(acc: str, form: str, day: str) -> bytes:
    return (f"<html><body>NVIDIA CORP Form {form} "
            f"Accession Number {acc} Filing Date {day} "
            f"Accepted {day} 16:21:23 "
            f"EX-99.1</body></html>").encode()


def _cohort() -> dict:
    entries = [(ACC1, "8-K", "2018-02-08"), (ACC2, "10-Q", "2019-03-18")]
    return {
        "schemaVersion": "sec_pit_availability_research_cohort_v1",
        "status": "SEC_PUBLIC_AVAILABILITY_NOT_INDEPENDENTLY_OBSERVED",
        "ticker": "NVDA", "benchmark": "SPY",
        "selectionRule": "CALENDAR_YEAR_FORM_FIRST_MIDDLE_LAST_NO_PRICE_RANKING",
        "priceOutcomesUsedForSelection": False,
        "secReceiptClockIsNotPublicAvailability": True,
        "trainingReady": False, "predictionReady": False,
        "modelPromotionAllowed": False,
        "knownFirstPublicAvailabilityCases": 0,
        "reviewQueueCount": 2, "filingCandidates": 2,
        "reviewQueue": [{
            "accession": acc, "form": form, "filingDate": day,
            "secIndexUrl": _url(acc), "selectedWithoutFutureReturns": True,
            "alreadyRawPinned": acc == ACC1,
        } for acc, form, day in entries],
        "filingAvailabilityLedger": [{
            "accession": acc, "form": form, "filingDate": day,
            "secIndexUrl": _url(acc),
            "secIndexSha256": hashlib.sha256(_index(acc, form, day)).hexdigest()
            if acc == ACC1 else None,
            "publicAvailabilityVerifiedByCutoff": False,
            "usableAsPreopenModelFeature": False,
            "canBeUsedAsNoNewsControl": False,
        } for acc, form, day in entries],
    }


def _client(seen: list[str]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        acc = ACC1 if str(request.url) == _url(ACC1) else ACC2
        form = "8-K" if acc == ACC1 else "10-Q"
        day = "2018-02-08" if acc == ACC1 else "2019-03-18"
        return httpx.Response(200, content=_index(acc, form, day))
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_capture_reuses_sha_pinned_and_only_fetches_missing(tmp_path: Path) -> None:
    original = tmp_path / f"sec-{ACC1}-index.html"
    original.write_bytes(_index(ACC1, "8-K", "2018-02-08"))
    seen: list[str] = []
    with _client(seen) as client:
        report = capture_sec_pit_evidence(
            cohort=_cohort(), cohort_sha="f" * 64,
            source_dir=tmp_path, limit=2,
            client=client, user_agent="Research contact@example.com",
            persist_raw=True,
        )
    assert seen == [_url(ACC2)]
    assert report["capturedIndexCount"] == 2
    assert report["newOriginalIndexCount"] == 1
    assert report["previouslyCachedCount"] == 1
    assert report["knownFirstPublicAvailabilityCases"] == 0
    assert report["trainingReady"] is False
    assert (tmp_path / f"sec-{ACC2}-index.html").exists()
    for entry in report["sourceRecords"]:
        assert entry["historicalFirstPublicAvailabilityAt"] is None
        assert entry["firstIndependentHistoricalObservedAt"] is None
        assert entry["verifiedPreopenMarketAvailability"] is False
        assert entry["eligibleForNewsModelTraining"] is False
        assert len(entry["secIndexSha256"]) == 64


def test_cached_offline_and_mutated_pinned_source_fails(tmp_path: Path) -> None:
    for acc, form, day in ((ACC1, "8-K", "2018-02-08"),
                           (ACC2, "10-Q", "2019-03-18")):
        (tmp_path / f"sec-{acc}-index.html").write_bytes(_index(acc, form, day))
    result = capture_sec_pit_evidence(
        cohort=_cohort(), cohort_sha="f" * 64, source_dir=tmp_path, limit=2,
    )
    assert result["newOriginalIndexCount"] == 0
    first = tmp_path / f"sec-{ACC1}-index.html"
    first.write_bytes(first.read_bytes().replace(b"16:21:23", b"16:21:24"))
    with pytest.raises(ValueError, match="SEC_SOURCE_SHA_MISMATCH"):
        capture_sec_pit_evidence(
            cohort=_cohort(), cohort_sha="f" * 64, source_dir=tmp_path, limit=2,
        )


def test_offline_uncached_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="SEC_SOURCE_NOT_YET_CAPTURED"):
        capture_sec_pit_evidence(
            cohort=_cohort(), cohort_sha="f" * 64, source_dir=tmp_path, limit=2,
        )


def test_provenance_and_wrong_issuer_path_rejected() -> None:
    data = _cohort()
    assert len(_validate_cohort(data)) == 2
    for key, value in (("trainingReady", True),
                       ("knownFirstPublicAvailabilityCases", 1),
                       ("priceOutcomesUsedForSelection", True)):
        changed = copy.deepcopy(data)
        changed[key] = value
        with pytest.raises(ValueError, match="SEC_COHORT_PROVENANCE_INVALID"):
            _validate_cohort(changed)
    assert _url_allowed(_url(ACC2), ACC2)
    assert not _url_allowed(
        "https://www.sec.gov/Archives/edgar/data/9999999/"
        + ACC2.replace("-", "") + "/" + ACC2 + "-index.htm", ACC2,
    )
    changed = copy.deepcopy(data)
    changed["reviewQueue"][1]["secIndexUrl"] = _url(ACC1)
    with pytest.raises(ValueError, match="SEC_COHORT_TASK_PROVENANCE_INVALID"):
        _validate_cohort(changed)


def test_no_backdated_receipt_or_unpinned_content(tmp_path: Path) -> None:
    raw = _index(ACC1, "8-K", "2018-02-08")
    (tmp_path / f"sec-{ACC1}-index.html").write_bytes(raw)
    changed = _cohort()
    changed["reviewQueue"] = changed["reviewQueue"][:1]
    changed["reviewQueueCount"] = 1
    changed["filingAvailabilityLedger"][0]["usableAsPreopenModelFeature"] = True
    with pytest.raises(ValueError, match="SEC_LEDGER_UNAUTHORIZED_PROMOTION"):
        capture_sec_pit_evidence(
            cohort=changed, cohort_sha="f" * 64,
            source_dir=tmp_path, limit=1,
        )
    changed = _cohort()
    changed["reviewQueue"] = changed["reviewQueue"][:1]
    changed["reviewQueueCount"] = 1
    (tmp_path / f"sec-{ACC1}-index.html").write_bytes(
        raw.replace(b"Accepted", b"Received")
    )
    with pytest.raises(ValueError, match="SEC_RAW_ACCEPTED_HEADER_MISSING"):
        capture_sec_pit_evidence(
            cohort=changed, cohort_sha="f" * 64,
            source_dir=tmp_path, limit=1,
        )
