"""SEC claim review rows require authentic pinned source and stay unreviewed."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
from test_sec_primary_semantic_review import _packet, _Provider

from tris_market_event_intelligence.sec_claim_evidence_review_cli import (
    build_claim_review,
)
from tris_market_event_intelligence.sec_primary_semantic_research_cli import (
    prepare_primary_cases,
    run_semantic_research,
)


def _inputs(tmp_path: Path) -> tuple[dict, list[tuple[dict, str]]]:
    primary = _packet(tmp_path, count=1)
    prepared = prepare_primary_cases(primary, "a" * 64, tmp_path)
    report = run_semantic_research(
        prepared, _Provider(prepared["cases"]), limit=1,
    )
    return primary, [(report, "b" * 64)]


def test_claim_packet_maps_all_fields_without_fake_truth_labels(tmp_path: Path) -> None:
    primary, reports = _inputs(tmp_path)
    result = build_claim_review(primary, "a" * 64, tmp_path, reports)
    assert result["claimCount"] == 8  # one observation, six hypothesis fields, summary
    assert result["casesIncluded"] == 1
    assert result["humanDecisionsRecorded"] == 0
    assert result["automatedSemanticEntailmentVerified"] is False
    assert result["trainingReady"] is False
    assert result["promotionAllowed"] is False
    assert len({c["location"] for c in result["claims"]}) == 8
    assert all(c["evidenceStatus"] == "NOT_ASSESSED" for c in result["claims"])
    assert all(c["humanEvidenceSupported"] is None for c in result["claims"])
    observation = next(x for x in result["claims"] if x["claimCategory"] == "ISSUER_OBSERVATION")
    assert observation["candidateEvidence"]
    assert all(
        item["exactSourceQuote"] in observation["sourceExcerpt"]
        for item in observation["candidateEvidence"]
    )
    sector = next(x for x in result["claims"] if x["claimCategory"] == "SECTOR_ALTERNATIVE")
    assert sector["candidateEvidence"] == []
    assert sector["independentIndustryEvidenceAvailable"] is False


def test_claim_packet_rejects_changed_original_source(tmp_path: Path) -> None:
    primary, reports = _inputs(tmp_path)
    accession = primary["cases"][0]["accession"]
    raw = tmp_path / f"sec-{accession}-primary.source"
    raw.write_bytes(raw.read_bytes().replace(b"CORPORATION", b"FABRICATED"))
    with pytest.raises(ValueError, match="SEC_SEMANTIC_SOURCE_SHA_MISMATCH"):
        build_claim_review(primary, "a" * 64, tmp_path, reports)


def test_claim_packet_rejects_model_fabricated_quote(tmp_path: Path) -> None:
    primary, reports = _inputs(tmp_path)
    bad = copy.deepcopy(reports[0][0])
    bad["cases"][0]["interpretation"]["observations"][0]["verbatim_quote"] = (
        "This claim does not occur in the pinned primary source."
    )
    with pytest.raises(ValueError, match="SEC_SEMANTIC_QUOTE_NOT_IN_PINNED_SOURCE"):
        build_claim_review(primary, "a" * 64, tmp_path, [(bad, "b" * 64)])


def test_claim_packet_rejects_historical_publication_upgrade(tmp_path: Path) -> None:
    primary, reports = _inputs(tmp_path)
    tampered = copy.deepcopy(primary)
    tampered["cases"][0]["trainingEligible"] = True
    with pytest.raises(ValueError, match="SEC_SEMANTIC_CASE_PROVENANCE_INVALID"):
        build_claim_review(tampered, "a" * 64, tmp_path, reports)
