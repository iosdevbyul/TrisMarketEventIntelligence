"""SEC primary semantic analysis uses original source bytes, never price outcomes."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tris_market_event_intelligence.sec_primary_content_research_cli import (
    _source_review,
)
from tris_market_event_intelligence.sec_primary_semantic_research_cli import (
    _validate_response,
    prepare_primary_cases,
    run_semantic_research,
)

ACC = "0001045810-18-000004"
BODY = (
    b"<html><body><h1>NVIDIA CORPORATION</h1>"
    b"<p>Item 2.02. Results of Operations and Financial Condition.</p>"
    b"<p>NVIDIA announced results in a filed press release."
    b" The quarterly revenue information is an issuer statement, "
    b"not independent verified evidence of future stock performance.</p>"
    b"</body></html>"
)
URL = (
    "https://www.sec.gov/Archives/edgar/data/1045810/"
    "000104581018000004/nvda-report.htm"
)
SHA = hashlib.sha256(BODY).hexdigest()


def _original_packet(directory: Path) -> dict:
    (directory / f"sec-{ACC}-primary.source").write_bytes(BODY)
    details = _source_review(BODY, URL)
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
        "reviewedIndexCases": 1,
        "primaryContentsPinned": 1,
        "cases": [{
            "accession": ACC,
            "form": "8-K",
            "filingDate": "2018-02-08",
            "status": "PRIMARY_DOCUMENT_SHA_PINNED_RESEARCH_ONLY",
            "primaryDocumentSha256": SHA,
            "primaryDocumentUrl": URL,
            "historicalPublicationAtVerified": False,
            "marketAvailabilityVerified": False,
            "usableAsPredecisionModelInput": False,
            "trainingEligible": False,
            **details,
        }],
    }


def _response(case: dict) -> dict:
    excerpt = case["evidence"][0]["text"]
    return {
        "accession": ACC,
        "source_sha256": SHA,
        "observations": [{
            "evidence_id": f"sec-{ACC}-primary",
            "verbatim_quote": "NVIDIA CORPORATION",
            "interpretation": "The original filing identifies NVIDIA as its issuer.",
        }, {
            "evidence_id": f"sec-{ACC}-primary",
            "verbatim_quote": "Results of Operations and Financial Condition",
            "interpretation": "The source labels a section covering operations.",
        }],
        "hypotheses": [
            {
                "category": "ISSUER_FUNDAMENTALS",
                "evidence_ids": [f"sec-{ACC}-primary"],
                "mechanism": "Issuer results might relate to fundamentals.",
                "limitation": "No independent verification of market reaction.",
            },
            {
                "category": "SECTOR_ALTERNATIVE",
                "evidence_ids": [],
                "mechanism": "Sector conditions might offer an alternative.",
                "limitation": "No independent sector facts were supplied here.",
            },
            {
                "category": "MISSING_CONTEXT",
                "evidence_ids": [],
                "mechanism": "Additional market context could change the assessment.",
                "limitation": "No independent historical dissemination proof.",
            },
        ],
        "summary": "The document is retrospective issuer evidence only.",
        "historical_public_time_verified": False,
        "causal_impact_proven": False,
        "stock_direction_predicted": False,
    }


class _StubProvider:
    model_id = "test-structured-runtime"

    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.messages: list[dict] = []

    def generate_json(self, *, system: str, user: str,
                      schema: dict[str, object] | None) -> str:
        self.messages.append(json.loads(user))
        assert "priceReturnsAvailable" in self.messages[-1]
        assert self.messages[-1]["priceReturnsAvailable"] is False
        assert "sourceEvidence" in self.messages[-1]
        assert schema is not None
        return json.dumps(self.responses.pop(0))


def test_prepare_exact_pinned_raw_bytes_not_just_json_excerpt(tmp_path: Path) -> None:
    packet = _original_packet(tmp_path)
    prepared = prepare_primary_cases(packet, "a" * 64, tmp_path)
    assert prepared["caseCount"] == 1
    assert prepared["cases"][0]["sourceSha256"] == SHA
    assert prepared["historicalPublicTimesIndependentlyVerified"] == 0
    assert prepared["stockOutcomeLabelsPresent"] is False
    assert prepared["trainingReady"] is False
    assert "NVIDIA CORPORATION" in prepared["cases"][0]["evidence"][0]["text"]


def test_model_inference_quote_citation_and_alternatives(tmp_path: Path) -> None:
    packet = _original_packet(tmp_path)
    prepared = prepare_primary_cases(packet, "a" * 64, tmp_path)
    stub = _StubProvider([_response(prepared["cases"][0])])
    report = run_semantic_research(prepared, stub, limit=1)
    assert len(stub.messages) == 1
    assert report["casesAnalyzed"] == 1
    assert report["allQuoteSpansVerifiedAgainstOriginalSHA"] is True
    assert report["modelCalls"] == 1
    assert report["stockOutcomesConsumed"] is False
    assert report["trainingReady"] is False
    assert report["historicalPublicTimesIndependentlyVerified"] == 0
    assert report["cases"][0]["interpretation"]["hypotheses"][1]["evidence_ids"] == []


def test_fabricated_quote_first_public_clock_and_sector_citation_rejected(
    tmp_path: Path,
) -> None:
    prepared = prepare_primary_cases(_original_packet(tmp_path), "a" * 64, tmp_path)
    case = prepared["cases"][0]
    bad_quote = _response(case)
    bad_quote["observations"][0]["verbatim_quote"] = (
        "NVIDIA forecast revenue of one trillion dollars."
    )
    with pytest.raises(ValueError, match="SEC_SEMANTIC_QUOTE_NOT_IN_PINNED_SOURCE"):
        _validate_response(case, json.dumps(bad_quote))
    invented_clock = _response(case)
    invented_clock["historical_public_time_verified"] = True
    with pytest.raises(ValueError, match="SEC_SEMANTIC_UNSUPPORTED_MODEL_CLAIM"):
        _validate_response(case, json.dumps(invented_clock))
    invented_sector = _response(case)
    invented_sector["hypotheses"][1]["evidence_ids"] = [f"sec-{ACC}-primary"]
    with pytest.raises(ValueError, match="SEC_SEMANTIC_HYPOTHESIS_UNSUPPORTED"):
        _validate_response(case, json.dumps(invented_sector))


def test_raw_source_mutation_and_excerpt_mismatch_rejected(tmp_path: Path) -> None:
    packet = _original_packet(tmp_path)
    path = tmp_path / f"sec-{ACC}-primary.source"
    path.write_bytes(BODY.replace(b"CORPORATION", b"INCORPORATED"))
    with pytest.raises(ValueError, match="SEC_SEMANTIC_SOURCE_SHA_MISMATCH"):
        prepare_primary_cases(packet, "a" * 64, tmp_path)
    path.write_bytes(BODY)
    changed = copy.deepcopy(packet)
    changed["cases"][0]["primarySourceVerbatimExcerpt"] = "NVIDIA invented information"
    with pytest.raises(ValueError, match="SEC_SEMANTIC_SOURCE_EXCERPT_MISMATCH"):
        prepare_primary_cases(changed, "a" * 64, tmp_path)


def test_source_pit_promotion_is_rejected(tmp_path: Path) -> None:
    packet = _original_packet(tmp_path)
    for field, value in (
        ("historicalPublicationAtVerified", True),
        ("marketAvailabilityVerified", True),
        ("trainingEligible", True),
    ):
        changed = copy.deepcopy(packet)
        changed["cases"][0][field] = value
        with pytest.raises(ValueError, match="SEC_SEMANTIC_CASE_PROVENANCE_INVALID"):
            prepare_primary_cases(changed, "a" * 64, tmp_path)
    changed = copy.deepcopy(packet)
    changed["trainingReady"] = True
    with pytest.raises(ValueError, match="SEC_SEMANTIC_PRIMARY_PROVENANCE_INVALID"):
        prepare_primary_cases(changed, "a" * 64, tmp_path)
