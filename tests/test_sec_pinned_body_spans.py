"""Raw-SHA-verified body selection, cite provenance and legacy comparability."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
from test_sec_primary_semantic_review import _packet, _Provider

from tris_market_event_intelligence.sec_claim_evidence_review_cli import (
    build_claim_review,
)
from tris_market_event_intelligence.sec_pinned_body_spans import (
    CONTEXT_VERSION,
    apply_body_spans,
)
from tris_market_event_intelligence.sec_primary_content_research_cli import (
    _source_review,
)
from tris_market_event_intelligence.sec_primary_semantic_research_cli import (
    prepare_primary_cases,
    run_semantic_research,
)
from tris_market_event_intelligence.sec_primary_semantic_review_cli import (
    build_review_packet,
)
from tris_market_event_intelligence.sec_semantic_evidence_gate import (
    SUBSTANTIVE,
    source_evidence_status,
)
from tris_market_event_intelligence.sec_semantic_paired_comparison_cli import (
    compare_semantic_reports,
)


def _full_body(directory: Path, *, risk: bool = False) -> dict:
    primary = _packet(directory, count=1)
    case = primary["cases"][0]
    cover = "NVIDIA CORPORATION. " + (
        "FORM 10-K ANNUAL REPORT PURSUANT TO SECTION 13. " * 27
    )
    material = (
        "NVIDIA relies on a limited number of suppliers for "
        "production components in its manufacturing process."
        if risk else
        "NVIDIA reported revenue of $5.2 billion during the fiscal period."
    )
    raw = ("<html><body>" + cover + "<p>" + material
           + "</p></body></html>").encode()
    source = directory / f"sec-{case['accession']}-primary.source"
    source.write_bytes(raw)
    case["primaryDocumentSha256"] = hashlib.sha256(raw).hexdigest()
    case.update(_source_review(raw, case["primaryDocumentUrl"]))
    assert len(case["primarySourceVerbatimExcerpt"]) == 1100
    assert material not in case["primarySourceVerbatimExcerpt"]
    return primary


def _candidate(tmp_path: Path) -> tuple[dict, dict, dict]:
    primary = _full_body(tmp_path)
    old_prepared = prepare_primary_cases(primary, "a" * 64, tmp_path)
    old = run_semantic_research(
        old_prepared, _Provider(old_prepared["cases"]), limit=1,
    )
    prepared = apply_body_spans(old_prepared, tmp_path)
    provider = _Provider(prepared["cases"])
    candidate = run_semantic_research(
        prepared, provider, limit=1, evidence_gated=True,
    )
    assert len(provider.calls) == 1
    return primary, old, candidate


def test_full_original_body_selected_after_1100_char_cover(tmp_path: Path) -> None:
    primary, old, candidate = _candidate(tmp_path)
    case = candidate["cases"][0]
    assert candidate["bodySpanSelectionVersion"] == CONTEXT_VERSION
    assert candidate["casesAbstained"] == 0
    assert case["analysisStatus"] == SUBSTANTIVE
    span = case["sourceSpan"]
    assert span["visibleStart"] >= 1100
    assert span["visibleEnd"] > span["visibleStart"]
    assert span["originalSourceSha256"] == primary["cases"][0]["primaryDocumentSha256"]
    assert len(span["visibleTextSha256"]) == 64
    review = build_review_packet(primary, "a" * 64, tmp_path, [(candidate, "c" * 64)])
    entry = review["cases"][0]
    assert entry["sourceSpan"] == span
    assert "NVIDIA reported revenue of $5.2 billion" in entry["sourceExcerpt"]
    assert entry["observations"][0]["verbatim_quote"] in entry["sourceExcerpt"]
    claims = build_claim_review(primary, "a" * 64, tmp_path, [(candidate, "c" * 64)])
    assert claims["claimCount"] == 8
    assert "NVIDIA reported revenue" in claims["claims"][0]["sourceExcerpt"]
    diff = compare_semantic_reports(
        primary, "a" * 64, tmp_path, [(old, "b" * 64)], [(candidate, "c" * 64)],
    )
    assert diff["pairs"][0]["evidenceScopeChanged"] is True
    assert diff["pairs"][0]["candidateSourceSpanSha256"] == span["spanSha256"]
    assert diff["semanticQualityImprovementVerified"] is False


def test_original_sha_and_exact_offsets_revalidated(tmp_path: Path) -> None:
    primary, _, candidate = _candidate(tmp_path)
    tampered = copy.deepcopy(candidate)
    tampered["cases"][0]["sourceSpan"]["visibleStart"] += 1
    with pytest.raises(ValueError, match="SEC_BODY_SPAN_PROVENANCE_INVALID"):
        build_review_packet(primary, "a" * 64, tmp_path, [(tampered, "c" * 64)])
    missing = copy.deepcopy(candidate)
    del missing["cases"][0]["sourceSpan"]
    with pytest.raises(ValueError, match="SEC_BODY_SPAN_SELECTION_MISMATCH"):
        build_review_packet(primary, "a" * 64, tmp_path, [(missing, "c" * 64)])
    raw_path = tmp_path / f"sec-{primary['cases'][0]['accession']}-primary.source"
    raw_path.write_bytes(raw_path.read_bytes().replace(b"revenue", b"fabric"))
    with pytest.raises(ValueError, match="SEC_SEMANTIC_SOURCE_SHA_MISMATCH"):
        build_review_packet(primary, "a" * 64, tmp_path, [(candidate, "c" * 64)])


def test_real_business_risk_prose_is_eligible_not_market_eligible(tmp_path: Path) -> None:
    primary = _full_body(tmp_path, risk=True)
    before = prepare_primary_cases(primary, "a" * 64, tmp_path)
    selected = apply_body_spans(before, tmp_path)
    case = selected["cases"][0]
    assert "sourceSpan" in case
    assert source_evidence_status(case["evidence"][0]["text"]) == SUBSTANTIVE
    assert case["historicalFirstPublicAt"] is None
    assert case["marketAvailableBeforeHistoricalCutoff"] is False
    assert selected["trainingReady"] is False


def test_cover_only_returns_no_body_source_and_no_generated_hypotheses(
    tmp_path: Path,
) -> None:
    primary = _packet(tmp_path, count=1)
    prepared = apply_body_spans(
        prepare_primary_cases(primary, "a" * 64, tmp_path), tmp_path,
    )
    assert prepared["bodySpanCasesSelected"] == 0
    provider = _Provider(prepared["cases"])
    report = run_semantic_research(prepared, provider, limit=1, evidence_gated=True)
    assert provider.calls == []
    assert report["modelCalls"] == 0
    assert report["casesAbstained"] == 1
    assert "sourceSpan" not in report["cases"][0]
    reviewed = build_review_packet(primary, "a" * 64, tmp_path, [(report, "c" * 64)])
    assert reviewed["cases"][0]["observations"] == []
