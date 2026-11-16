"""Offline first-case human review and bounded offset studies with real SHA fixtures."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tris_market_event_intelligence.sec_primary_content_research_cli import _source_review
from tris_market_event_intelligence.sec_primary_semantic_quality_cli import (
    audit_semantic_quality,
)
from tris_market_event_intelligence.sec_primary_semantic_research_cli import (
    _quote_anchors,
    prepare_primary_cases,
    run_semantic_research,
)
from tris_market_event_intelligence.sec_primary_semantic_review_cli import (
    build_review_packet,
)

BODY = (
    b"<html><body><h1>NVIDIA CORPORATION</h1>"
    b"<p>Item 2.02. Results of Operations and Financial Condition.</p>"
    b"<p>NVIDIA announced results in a filed press release. "
    b"These results do not establish stock returns or market impact.</p>"
    b"</body></html>"
)
SHA = hashlib.sha256(BODY).hexdigest()


def _packet(directory: Path, count: int = 3) -> dict:
    cases = []
    for i in range(count):
        acc = f"0001045810-18-{i + 4:06d}"
        url = (
            "https://www.sec.gov/Archives/edgar/data/1045810/"
            f"{acc.replace('-', '')}/nvda-report.htm"
        )
        (directory / f"sec-{acc}-primary.source").write_bytes(BODY)
        cases.append({
            "accession": acc,
            "form": "8-K",
            "filingDate": "2018-02-08",
            "status": "PRIMARY_DOCUMENT_SHA_PINNED_RESEARCH_ONLY",
            "primaryDocumentSha256": SHA,
            "primaryDocumentUrl": url,
            "historicalPublicationAtVerified": False,
            "marketAvailabilityVerified": False,
            "usableAsPredecisionModelInput": False,
            "trainingEligible": False,
            **_source_review(BODY, url),
        })
    return {
        "schemaVersion": "sec_original_primary_content_research_v1",
        "status": "SEC_ORIGINAL_CONTENT_WITH_UNVERIFIED_HISTORICAL_PUBLICATION",
        "ticker": "NVDA",
        "originalSourceTextOnly": True,
        "independentHistoricalPublicationTimesVerified": 0,
        "outcomeDataRead": False,
        "modelCalls": 0,
        "trainingReady": False,
        "promotionAllowed": False,
        "reviewedIndexCases": len(cases),
        "primaryContentsPinned": len(cases),
        "cases": cases,
    }


def _model(case: dict) -> dict:
    return {
        "accession": case["accession"],
        "source_sha256": case["sourceSha256"],
        "observations": [{
            "evidence_id": case["evidence"][0]["id"],
            "verbatim_quote": "NVIDIA CORPORATION",
            "interpretation": "The original SEC document identifies NVIDIA as issuer.",
        }],
        "hypotheses": [
            {
                "category": "ISSUER_FUNDAMENTALS", "evidence_ids": [],
                "mechanism": "Issuer statements might relate to company fundamentals.",
                "limitation": "These statements are not proof of any market effect.",
            },
            {
                "category": "SECTOR_ALTERNATIVE", "evidence_ids": [],
                "mechanism": "Unknown industry conditions could be an alternative.",
                "limitation": "No independent industry observations were supplied.",
            },
            {
                "category": "MISSING_CONTEXT", "evidence_ids": [],
                "mechanism": "Unobserved contextual factors could matter instead.",
                "limitation": "The historical publication timestamp is unverified.",
            },
        ],
        "summary": "The excerpt is issuer content, not market evidence.",
        "historical_public_time_verified": False,
        "causal_impact_proven": False,
        "stock_direction_predicted": False,
    }


class _Provider:
    model_id = "fixed-test-provider"

    def __init__(self, cases: list[dict]) -> None:
        self.by_acc = {c["accession"]: c for c in cases}
        self.calls: list[str] = []

    def generate_json(self, *, system: str, user: str,
                      schema: dict | None) -> str:
        acc = json.loads(user)["accession"]
        self.calls.append(acc)
        anchored = _model(self.by_acc[acc])
        for key in (
            "accession", "source_sha256",
            "historical_public_time_verified", "causal_impact_proven",
            "stock_direction_predicted",
        ):
            del anchored[key]
        quote_id = next(iter(_quote_anchors(self.by_acc[acc])))
        anchored["observations"] = [{
            "quote_id": quote_id,
            "interpretation": anchored["observations"][0]["interpretation"],
        }]
        return json.dumps(anchored)


def test_offset_batch_preserves_original_case_indices(tmp_path: Path) -> None:
    primary = _packet(tmp_path)
    prepared = prepare_primary_cases(primary, "a" * 64, tmp_path)
    provider = _Provider(prepared["cases"])
    report = run_semantic_research(prepared, provider, limit=2, start_index=1)
    assert report["startIndex"] == 1
    assert report["originalCasesAvailable"] == 3
    assert [c["caseIndex"] for c in report["cases"]] == [1, 2]
    assert provider.calls == [c["accession"] for c in prepared["cases"][1:]]
    assert report["modelCalls"] == 2
    assert report["trainingReady"] is False


@pytest.mark.parametrize("index", [-1, 3, True])
def test_invalid_offsets_rejected_without_model_call(
    tmp_path: Path, index: int | bool,
) -> None:
    prepared = prepare_primary_cases(_packet(tmp_path), "a" * 64, tmp_path)
    provider = _Provider(prepared["cases"])
    with pytest.raises(ValueError, match="SEC_SEMANTIC_MODEL_INPUT_INVALID"):
        run_semantic_research(prepared, provider, start_index=index)
    assert not provider.calls


def test_report_reverification_and_pending_human_review(tmp_path: Path) -> None:
    primary = _packet(tmp_path)
    prepared = prepare_primary_cases(primary, "a" * 64, tmp_path)
    report = run_semantic_research(
        prepared, _Provider(prepared["cases"]), limit=1, start_index=1,
    )
    packet = build_review_packet(primary, "a" * 64, tmp_path, [(report, "b" * 64)])
    assert packet["reviewCasesPrepared"] == 1
    assert packet["totalPinnedCases"] == 3
    assert [x["caseIndex"] for x in packet["pendingCaseIndexes"]] == [0, 2]
    assert packet["allSECQuotesReverifiedAgainstPinnedSources"] is True
    assert packet["cases"][0]["observations"][0]["humanQuoteAccurate"] is None
    assert packet["cases"][0]["hypotheses"][0]["humanReasonablePossibility"] is None
    assert packet["cases"][0]["humanReviewComplete"] is False
    assert packet["trainingReady"] is False


def test_legacy_firstcase_without_offset_is_accepted(tmp_path: Path) -> None:
    primary = _packet(tmp_path)
    prepared = prepare_primary_cases(primary, "a" * 64, tmp_path)
    report = run_semantic_research(prepared, _Provider(prepared["cases"]), limit=1)
    del report["cases"][0]["caseIndex"]
    del report["startIndex"]
    del report["originalCasesAvailable"]
    packet = build_review_packet(primary, "a" * 64, tmp_path, [(report, "c" * 64)])
    assert packet["cases"][0]["caseIndex"] == 0


def test_duplicate_and_report_provenance_rejected(tmp_path: Path) -> None:
    primary = _packet(tmp_path)
    prepared = prepare_primary_cases(primary, "a" * 64, tmp_path)
    report = run_semantic_research(prepared, _Provider(prepared["cases"]), limit=1)
    with pytest.raises(ValueError, match="SEC_SEMANTIC_REVIEW_DUPLICATE_OR_UNKNOWN_CASE"):
        build_review_packet(
            primary, "a" * 64, tmp_path, [(report, "b" * 64), (report, "c" * 64)],
        )
    bad = copy.deepcopy(report)
    bad["historicalPublicTimesIndependentlyVerified"] = 1
    with pytest.raises(ValueError, match="SEC_SEMANTIC_REVIEW_REPORT_PROVENANCE_INVALID"):
        build_review_packet(primary, "a" * 64, tmp_path, [(bad, "b" * 64)])
    bad = copy.deepcopy(report)
    bad["modelCalls"] = 8
    with pytest.raises(ValueError, match="SEC_SEMANTIC_REVIEW_MODEL_CALL_COUNT_INVALID"):
        build_review_packet(primary, "a" * 64, tmp_path, [(bad, "b" * 64)])


def test_mutated_local_original_or_forged_quote_rejected(tmp_path: Path) -> None:
    primary = _packet(tmp_path)
    prepared = prepare_primary_cases(primary, "a" * 64, tmp_path)
    report = run_semantic_research(prepared, _Provider(prepared["cases"]), limit=1)
    bad = copy.deepcopy(report)
    bad["cases"][0]["interpretation"]["observations"][0]["verbatim_quote"] = (
        "Fabricated source observation"
    )
    with pytest.raises(ValueError, match="SEC_SEMANTIC_QUOTE_NOT_IN_PINNED_SOURCE"):
        build_review_packet(primary, "a" * 64, tmp_path, [(bad, "b" * 64)])
    path = tmp_path / f"sec-{primary['cases'][0]['accession']}-primary.source"
    path.write_bytes(BODY.replace(b"CORPORATION", b"LABORATORY"))
    with pytest.raises(ValueError, match="SEC_SEMANTIC_SOURCE_SHA_MISMATCH"):
        build_review_packet(primary, "a" * 64, tmp_path, [(report, "b" * 64)])


def test_semantic_quality_triage_reverifies_input_and_keeps_decisions_pending(
    tmp_path: Path,
) -> None:
    primary = _packet(tmp_path)
    prepared = prepare_primary_cases(primary, "a" * 64, tmp_path)
    report = run_semantic_research(prepared, _Provider(prepared["cases"]), limit=2)
    review = build_review_packet(primary, "a" * 64, tmp_path, [(report, "b" * 64)])
    audit = audit_semantic_quality(
        primary, "a" * 64, tmp_path, [(report, "b" * 64)], review, "c" * 64,
    )
    assert audit["casesAudited"] == 2
    assert audit["remainingCaseIndexes"] == [2]
    assert audit["heuristicFindingCount"] == 0
    assert audit["humanDecisionsRecorded"] == 0
    assert audit["zeroHeuristicFindingsDoesNotProveSemanticAccuracy"] is True
    assert audit["trainingReady"] is False
    assert audit["reviewApproved"] is False
    assert "sourceExcerpt" not in json.dumps(audit)
    assert "NVIDIA CORPORATION" not in json.dumps(audit)


def test_semantic_quality_flags_overclaims_without_leaking_unverified_prose(
    tmp_path: Path,
) -> None:
    primary = _packet(tmp_path)
    prepared = prepare_primary_cases(primary, "a" * 64, tmp_path)
    report = run_semantic_research(prepared, _Provider(prepared["cases"]), limit=1)
    interpretation = report["cases"][0]["interpretation"]
    interpretation["summary"] = "This news definitely caused the stock to rise."
    interpretation["hypotheses"][0]["mechanism"] = (
        "Investors knew about this issuer report before the market opened."
    )
    review = build_review_packet(primary, "a" * 64, tmp_path, [(report, "b" * 64)])
    audit = audit_semantic_quality(
        primary, "a" * 64, tmp_path, [(report, "b" * 64)], review, "c" * 64,
    )
    codes = {finding["code"] for finding in audit["heuristicFindings"]}
    assert "POTENTIAL_UNSUPPORTED_CAUSAL_LANGUAGE" in codes
    assert "POTENTIAL_UNVERIFIED_PUBLICATION_CLAIM" in codes
    assert "POTENTIAL_STOCK_DIRECTION_ASSERTION" not in codes
    assert all(set(finding) == {"caseIndex", "location", "code"}
               for finding in audit["heuristicFindings"])
    assert "Investors knew" not in json.dumps(audit)
    assert "definitely caused" not in json.dumps(audit)


def test_semantic_quality_refuses_stale_tampered_or_fake_review(
    tmp_path: Path,
) -> None:
    primary = _packet(tmp_path)
    prepared = prepare_primary_cases(primary, "a" * 64, tmp_path)
    report = run_semantic_research(prepared, _Provider(prepared["cases"]), limit=1)
    review = build_review_packet(primary, "a" * 64, tmp_path, [(report, "b" * 64)])
    changed = copy.deepcopy(review)
    changed["cases"][0]["humanReviewComplete"] = True
    with pytest.raises(ValueError, match="SEC_SEMANTIC_QUALITY_REVIEW_PACKET_MISMATCH"):
        audit_semantic_quality(
            primary, "a" * 64, tmp_path, [(report, "b" * 64)], changed, "c" * 64,
        )
    changed = copy.deepcopy(report)
    changed["cases"][0]["sourceSha256"] = "f" * 64
    with pytest.raises(ValueError, match="SEC_SEMANTIC_REVIEW_CASE_PROVENANCE_INVALID"):
        audit_semantic_quality(
            primary, "a" * 64, tmp_path, [(changed, "b" * 64)], review, "c" * 64,
        )
    with pytest.raises(ValueError, match="SEC_SEMANTIC_QUALITY_REVIEW_PACKET_MISMATCH"):
        audit_semantic_quality(
            primary, "a" * 64, tmp_path, [(report, "b" * 64)], review, "not-sha",
        )
