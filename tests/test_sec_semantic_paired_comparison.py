"""Paired SEC interpretation differences are not quality or truth labels."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
from test_sec_primary_semantic_review import _packet, _Provider

from tris_market_event_intelligence.sec_primary_semantic_research_cli import (
    prepare_primary_cases,
    run_semantic_research,
)
from tris_market_event_intelligence.sec_semantic_paired_comparison_cli import (
    compare_semantic_reports,
)


def _pair(tmp_path: Path) -> tuple[dict, dict, dict]:
    primary = _packet(tmp_path, count=2)
    prepared = prepare_primary_cases(primary, "a" * 64, tmp_path)
    original = run_semantic_research(prepared, _Provider(prepared["cases"]), limit=2)
    candidate = copy.deepcopy(original)
    candidate["cases"][0]["interpretation"]["summary"] = (
        "A different but still retrospective and human-unverified interpretation."
    )
    return primary, original, candidate


def test_same_source_comparison_reports_only_changes(tmp_path: Path) -> None:
    primary, original, candidate = _pair(tmp_path)
    result = compare_semantic_reports(
        primary, "a" * 64, tmp_path, [(original, "b" * 64)],
        [(candidate, "c" * 64)],
    )
    assert result["pairedCases"] == 2
    assert result["fieldsDifferent"] == 1
    assert result["status"] == "DIFFERENCE_ONLY_HUMAN_REVIEW_REQUIRED"
    assert result["humanDecisionsRecorded"] == 0
    assert result["semanticQualityImprovementVerified"] is False
    assert result["trainingReady"] is False
    assert "modelStatement" not in str(result)
    first = result["pairs"][0]
    assert any(x == {"location": "summary", "change": "CHANGED"} for x in first["fields"])


def test_reject_mismatched_case_set(tmp_path: Path) -> None:
    primary, original, candidate = _pair(tmp_path)
    candidate["cases"] = candidate["cases"][:1]
    candidate["casesAnalyzed"] = 1
    candidate["modelCalls"] = 1
    with pytest.raises(ValueError, match="SEC_PAIRED_CASE_SET_MISMATCH"):
        compare_semantic_reports(
            primary, "a" * 64, tmp_path, [(original, "b" * 64)],
            [(candidate, "c" * 64)],
        )


def test_reject_not_independent_candidate(tmp_path: Path) -> None:
    primary, original, _ = _pair(tmp_path)
    with pytest.raises(ValueError, match="SEC_PAIRED_INDEPENDENT_CANDIDATE_REQUIRED"):
        compare_semantic_reports(
            primary, "a" * 64, tmp_path, [(original, "b" * 64)],
            [(original, "b" * 64)],
        )


def test_reject_invalid_primary_bytes(tmp_path: Path) -> None:
    primary, original, candidate = _pair(tmp_path)
    accession = primary["cases"][0]["accession"]
    path = tmp_path / f"sec-{accession}-primary.source"
    path.write_bytes(path.read_bytes().replace(b"CORPORATION", b"FABRICATED"))
    with pytest.raises(ValueError, match="SEC_SEMANTIC_SOURCE_SHA_MISMATCH"):
        compare_semantic_reports(
            primary, "a" * 64, tmp_path, [(original, "b" * 64)],
            [(candidate, "c" * 64)],
        )


def test_reject_tampered_quote_in_candidate(tmp_path: Path) -> None:
    primary, original, candidate = _pair(tmp_path)
    candidate["cases"][0]["interpretation"]["observations"][0]["verbatim_quote"] = (
        "Nonexistent source words from fabricated evidence."
    )
    with pytest.raises(ValueError, match="SEC_SEMANTIC_QUOTE_NOT_IN_PINNED_SOURCE"):
        compare_semantic_reports(
            primary, "a" * 64, tmp_path, [(original, "b" * 64)],
            [(candidate, "c" * 64)],
        )
