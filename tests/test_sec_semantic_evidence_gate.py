"""Materiality abstention, source-specific claim checks and legacy interoperability."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
from test_sec_primary_semantic_review import _Provider, _packet

from tris_market_event_intelligence.sec_claim_evidence_review_cli import build_claim_review
from tris_market_event_intelligence.sec_primary_content_research_cli import _source_review
from tris_market_event_intelligence.sec_primary_semantic_research_cli import (
    prepare_primary_cases,
    run_semantic_research,
)
from tris_market_event_intelligence.sec_primary_semantic_review_cli import (
    build_review_packet,
)
from tris_market_event_intelligence.sec_semantic_evidence_gate import (
    INSUFFICIENT,
    SUBSTANTIVE,
    source_evidence_status,
    validate_concrete_claims,
)
from tris_market_event_intelligence.sec_semantic_paired_comparison_cli import (
    compare_semantic_reports,
)


def _material_packet(directory: Path) -> dict:
    primary = _packet(directory, 1)
    case = primary["cases"][0]
    raw = (
        b"<html><body><h1>NVIDIA CORPORATION</h1>"
        b"<p>NVIDIA reported revenue of $5.2 billion in the fiscal period. "
        b"The company reported gross margin of 52 percent in the same period. "
        b"This financial disclosure is retrospective primary issuer evidence.</p>"
        b"</body></html>"
    )
    path = directory / f"sec-{case['accession']}-primary.source"
    path.write_bytes(raw)
    case["primaryDocumentSha256"] = hashlib.sha256(raw).hexdigest()
    case.update(_source_review(raw, case["primaryDocumentUrl"]))
    return primary


def test_annual_header_abstains_without_calling_provider(tmp_path: Path) -> None:
    primary = _packet(tmp_path, 1)
    prepared = prepare_primary_cases(primary, "a" * 64, tmp_path)
    provider = _Provider(prepared["cases"])
    report = run_semantic_research(prepared, provider, evidence_gated=True, limit=1)
    assert provider.calls == []
    assert report["modelCalls"] == 0
    assert report["hypothesesCorrectionCount"] == 0
    assert report["casesAbstained"] == 1
    row = report["cases"][0]
    assert row["analysisStatus"] == INSUFFICIENT
    assert row["interpretation"]["observations"] == []
    assert row["interpretation"]["hypotheses"] == []
    review = build_review_packet(primary, "a" * 64, tmp_path, [(report, "b" * 64)])
    assert review["cases"][0]["analysisStatus"] == INSUFFICIENT
    assert review["modelCallsObserved"] == 0
    claim_review = build_claim_review(primary, "a" * 64, tmp_path, [(report, "b" * 64)])
    assert claim_review["claimCount"] == 0
    assert claim_review["abstainedCaseIndexes"] == [0]
    assert claim_review["humanDecisionsRecorded"] == 0


def test_material_source_allows_a_bounded_model_call(tmp_path: Path) -> None:
    primary = _material_packet(tmp_path)
    prepared = prepare_primary_cases(primary, "a" * 64, tmp_path)
    provider = _Provider(prepared["cases"])
    report = run_semantic_research(prepared, provider, evidence_gated=True, limit=1)
    assert report["modelCalls"] == 1
    assert report["casesAbstained"] == 0
    assert report["cases"][0]["analysisStatus"] == SUBSTANTIVE
    assert provider.calls == [prepared["cases"][0]["accession"]]
    review = build_review_packet(primary, "a" * 64, tmp_path, [(report, "b" * 64)])
    assert len(review["cases"][0]["hypotheses"]) == 3


@pytest.mark.parametrize("claim", [
    "This 10-K suggests underlying internal restructuring and executive oversight changes.",
    "The filing may indicate a shift in internal accounting practices.",
    "The sector changed because of semiconductor supply chain shifts.",
    "The missing context could involve a major product launch or lawsuit settlement.",
])
def test_unsupported_claim_fails_despite_a_material_revenue_excerpt(
    tmp_path: Path, claim: str,
) -> None:
    primary = _material_packet(tmp_path)
    prepared = prepare_primary_cases(primary, "a" * 64, tmp_path)
    case = prepared["cases"][0]
    interpretation = {
        "summary": "This is issuer-reported financial information.",
        "observations": [{"interpretation": "The filing reports revenue and margin data."}],
        "hypotheses": [
            {"category": "ISSUER_FUNDAMENTALS", "mechanism": claim,
             "limitation": "No independent causality is established."},
            {"category": "SECTOR_ALTERNATIVE",
             "mechanism": "Unknown external factors might be relevant.",
             "limitation": "Independent sector data were not provided."},
            {"category": "MISSING_CONTEXT",
             "mechanism": "Other filings may contain additional information.",
             "limitation": "The supplied source excerpt is incomplete."},
        ],
    }
    with pytest.raises(ValueError, match="SEC_SEMANTIC_"):
        validate_concrete_claims(case, interpretation)


def test_legacy_report_and_abstained_candidate_pair_without_fake_quality(
    tmp_path: Path,
) -> None:
    primary = _packet(tmp_path, 1)
    prepared = prepare_primary_cases(primary, "a" * 64, tmp_path)
    old = run_semantic_research(prepared, _Provider(prepared["cases"]), limit=1)
    new = run_semantic_research(
        prepared, _Provider(prepared["cases"]), limit=1, evidence_gated=True,
    )
    result = compare_semantic_reports(
        primary, "a" * 64, tmp_path,
        [(old, "b" * 64)], [(new, "c" * 64)],
    )
    assert result["fieldsDifferent"] > 0
    assert result["semanticQualityImprovementVerified"] is False
    assert result["humanDecisionsRecorded"] == 0
    assert any(
        change["location"] == "analysisStatus"
        for change in result["pairs"][0]["fields"]
    )


def test_forged_abstention_on_material_source_is_rejected(tmp_path: Path) -> None:
    primary = _material_packet(tmp_path)
    prepared = prepare_primary_cases(primary, "a" * 64, tmp_path)
    report = run_semantic_research(prepared, _Provider(prepared["cases"]), limit=1)
    forged = copy.deepcopy(report)
    forged["evidenceGateVersion"] = "sec_substantive_evidence_v1"
    forged["casesAbstained"] = 1
    forged["modelCalls"] = 0
    forged["cases"][0]["analysisStatus"] = INSUFFICIENT
    forged["cases"][0]["quoteAnchorSelectionApplied"] = False
    forged["cases"][0]["interpretation"] = {
        "observations": [], "hypotheses": [],
        "summary": (
            "The SHA-verified SEC excerpt provides filing identity or administrative "
            "context only. Issuer-event and sector hypotheses are withheld because "
            "the available excerpt does not substantiate a material business event."
        ),
    }
    with pytest.raises(ValueError, match="SEC_SEMANTIC_ABSTENTION_PROVENANCE_INVALID"):
        build_review_packet(primary, "a" * 64, tmp_path, [(forged, "b" * 64)])


def test_classifier_no_keyword_only_promotion() -> None:
    assert source_evidence_status(
        "FORM 10-K ANNUAL REPORT NVIDIA CORPORATION. "
        "Fiscal year ended 2017. Table of Contents Item 2.02."
    ) == INSUFFICIENT
    assert source_evidence_status(
        "NVIDIA reported revenue of $5.2 billion for the quarter. "
        "The company reported gross margin of 52 percent."
    ) == SUBSTANTIVE
