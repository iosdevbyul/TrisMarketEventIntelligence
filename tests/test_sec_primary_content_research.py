"""Ground SEC primary source facts, not publication clocks or price labels."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import httpx
import pytest

from tris_market_event_intelligence.sec_primary_content_research_cli import (
    _primary_url, _source_review, build_primary_evidence,
)

ACC = "0001045810-18-000004"
AGENT_ACC = "0001193125-19-000003"


def _url(acc: str, filename: str) -> str:
    return ("https://www.sec.gov/Archives/edgar/data/1045810/"
            + acc.replace("-", "") + "/" + filename)


def _index(acc: str, form: str, day: str, filename: str) -> bytes:
    return (
        f"<html><body>NVIDIA Form {form} Accession Number {acc}"
        f" Filing Date {day} Accepted {day} 16:21:23"
        '<table summary="Document Format Files">'
        f'<tr><td>1</td><td>Document</td><td><a href="{_url(acc, filename)}">'
        f"{filename}</a></td><td>{form}</td><td>123456</td></tr>"
        f'<tr><td>2</td><td>EX-99.1</td><td><a href="{_url(acc, "ex99.htm")}">'
        "ex99.htm</a></td><td>EX-99.1</td><td>9999</td></tr>"
        "</table></body></html>"
    ).encode()


def _content() -> bytes:
    return (
        "<html><body><h1>NVIDIA CORPORATION</h1>"
        "<h2>Item 2.02. Results of Operations and Financial Condition</h2>"
        "<p>NVIDIA announced its quarterly results in a press release. "
        "The reported information is provided only for historical source "
        "research and must not be used as a real-time trading signal.</p>"
        "</body></html>"
    ).encode()


def _fixtures(tmp_path: Path) -> tuple[dict, dict]:
    items = [(ACC, "8-K", "2018-02-08", "nvda-report.htm"),
             (AGENT_ACC, "10-Q", "2019-03-18", "quarterly.htm")]
    queue, ledger, captured = [], [], []
    for acc, form, day, file in items:
        index_bytes = _index(acc, form, day, file)
        (tmp_path / f"sec-{acc}-index.html").write_bytes(index_bytes)
        uri = _url(acc, acc + "-index.htm")
        queue.append({
            "accession": acc, "form": form,
            "filingDate": day, "secIndexUrl": uri,
            "selectedWithoutFutureReturns": True,
            "alreadyRawPinned": acc == ACC,
        })
        ledger.append({
            "accession": acc, "form": form,
            "filingDate": day, "secIndexUrl": uri,
            "secIndexSha256": hashlib.sha256(index_bytes).hexdigest()
            if acc == ACC else None,
            "publicAvailabilityVerifiedByCutoff": False,
            "usableAsPreopenModelFeature": False,
            "canBeUsedAsNoNewsControl": False,
        })
        captured.append({
            "accession": acc, "form": form,
            "secIndexSha256": hashlib.sha256(index_bytes).hexdigest(),
            "historicalFirstPublicAvailabilityAt": None,
            "eligibleForNewsModelTraining": False,
            "firstIndependentHistoricalObservedAt": None,
        })
    cohort = {
        "schemaVersion": "sec_pit_availability_research_cohort_v1",
        "status": "SEC_PUBLIC_AVAILABILITY_NOT_INDEPENDENTLY_OBSERVED",
        "ticker": "NVDA", "benchmark": "SPY",
        "selectionRule": "CALENDAR_YEAR_FORM_FIRST_MIDDLE_LAST_NO_PRICE_RANKING",
        "priceOutcomesUsedForSelection": False,
        "secReceiptClockIsNotPublicAvailability": True,
        "trainingReady": False, "predictionReady": False,
        "modelPromotionAllowed": False, "knownFirstPublicAvailabilityCases": 0,
        "filingCandidates": 2, "reviewQueueCount": 2,
        "reviewQueue": queue, "filingAvailabilityLedger": ledger,
    }
    batch = {
        "schemaVersion": "sec_pit_original_index_capture_v1",
        "status": "SEC_ORIGINAL_INDEX_SHA_PINNED_PUBLICATION_TIME_UNPROVEN",
        "cohortSha256": "a" * 64, "trainingReady": False,
        "knownFirstPublicAvailabilityCases": 0,
        "capturedIndexCount": 2,
        "sourceRecords": captured,
    }
    return cohort, batch


def test_real_sec_index_table_finds_form_primary_not_exhibit(tmp_path: Path) -> None:
    cohort, _ = _fixtures(tmp_path)
    task = cohort["reviewQueue"][0]
    raw = (tmp_path / f"sec-{ACC}-index.html").read_bytes()
    assert _primary_url(raw, task) == _url(ACC, "nvda-report.htm")
    changed = raw.replace(b"<td>8-K</td>", b"<td>EX-99.1</td>")
    assert _primary_url(changed, task) is None


def test_content_capture_and_semantic_source_excerpt(tmp_path: Path) -> None:
    cohort, batch = _fixtures(tmp_path)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=_content())

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = build_primary_evidence(
            cohort=cohort, cohort_sha="a" * 64,
            batch=batch, batch_sha="b" * 64,
            source_dir=tmp_path, limit=2,
            run=True, client=client, user_agent="Researcher dev@example.org",
        )
    assert seen == [_url(ACC, "nvda-report.htm"),
                    _url(AGENT_ACC, "quarterly.htm")]
    assert result["primaryContentsPinned"] == 2
    assert result["primaryContentsNew"] == 2
    assert result["primaryNotYetResolved"] == 0
    assert result["independentHistoricalPublicationTimesVerified"] == 0
    assert all(row["usableAsPredecisionModelInput"] is False
               for row in result["cases"])
    assert result["cases"][0]["secItemMarkersObservedInSource"] == ["ITEM2.02"]
    assert "NVIDIA CORPORATION" in result["cases"][0]["primarySourceVerbatimExcerpt"]
    assert result["modelCalls"] == 0
    assert result["outcomeDataRead"] is False
    assert (tmp_path / f"sec-{ACC}-primary.source").exists()
    second = build_primary_evidence(
        cohort=cohort, cohort_sha="a" * 64,
        batch=batch, batch_sha="b" * 64,
        source_dir=tmp_path, limit=2,
        run=False,
    )
    assert second["primaryContentsReused"] == 2
    assert second["primaryContentsNew"] == 0


def test_preview_does_not_fetch_or_claim_content_without_source(tmp_path: Path) -> None:
    cohort, batch = _fixtures(tmp_path)
    result = build_primary_evidence(
        cohort=cohort, cohort_sha="a" * 64,
        batch=batch, batch_sha="b" * 64,
        source_dir=tmp_path, limit=2,
    )
    assert result["primaryContentsPinned"] == 0
    assert result["primaryNotYetResolved"] == 2
    assert all(row["trainingEligible"] is False for row in result["cases"])


def test_source_sha_guard_and_training_block(tmp_path: Path) -> None:
    cohort, batch = _fixtures(tmp_path)
    changed = copy.deepcopy(batch)
    changed["sourceRecords"][0]["secIndexSha256"] = "e" * 64
    with pytest.raises(ValueError, match="SEC_PRIMARY_INDEX_SHA_INVALID"):
        build_primary_evidence(
            cohort=cohort, cohort_sha="a" * 64,
            batch=changed, batch_sha="b" * 64,
            source_dir=tmp_path, limit=2,
        )
    modified = copy.deepcopy(cohort)
    modified["trainingReady"] = True
    with pytest.raises(ValueError, match="SEC_COHORT_PROVENANCE_INVALID"):
        build_primary_evidence(
            cohort=modified, cohort_sha="a" * 64,
            batch=batch, batch_sha="b" * 64,
            source_dir=tmp_path, limit=2,
        )
    with pytest.raises(ValueError, match="SEC_PRIMARY_DOCUMENT_ISSUER_OR_CONTENT_INVALID"):
        _source_review(b"<html><body>unrelated</body></html>",
                       _url(ACC, "nvda-report.htm"))
