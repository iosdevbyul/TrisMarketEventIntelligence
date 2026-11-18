"""Temporal coverage and mode routing tests for live-source generalization."""
from __future__ import annotations

from datetime import date
from pathlib import Path

from test_market_evidence_db_bridge import populated_db

from tris_market_event_intelligence.market_pressure_generalization_cli import (
    evaluate_pages,
)


def test_multi_page_generalization_preview_is_source_only(tmp_path: Path) -> None:
    db = populated_db(tmp_path / "news.sqlite3")
    report = evaluate_pages(
        db, as_of=date(2024, 2, 15), lookback_days=14,
        pages=4, provider=None,
    )
    assert report["status"] == "READ_ONLY_COVERAGE_PREVIEW"
    assert report["uniqueEligibleSources"] == 2
    assert report["candidatePagesExamined"] == 1
    assert report["modelCalls"] == 0
    assert report["sourceValidatedImpactCount"] == 0
    assert report["generalizationVerified"] is False
    assert report["holdoutIndependenceCertified"] is False
    assert report["semanticAccuracyVerified"] is False
    assert report["databaseModified"] is False
    assert report["candidateSourcesActuallyGrouped"] == 0
    assert report["groupingCoverageFraction"] == 0.0
    assert report["samplingWarnings"] == [
        "PUBLISHER_DIVERSITY_NOT_ESTABLISHED"
    ]
    assert len(report["publisherCounts"]) == 2
    assert len(report["pages"][0]["sourceIds"]) == 2


def test_future_cutoff_does_not_enter_generalization_preview(
    tmp_path: Path,
) -> None:
    db = populated_db(tmp_path / "news.sqlite3")
    report = evaluate_pages(
        db, as_of=date(2024, 2, 1), lookback_days=30,
        pages=4, provider=None,
    )
    assert report["uniqueEligibleSources"] == 0
    assert report["modelCalls"] == 0
    assert report["eventGroupCount"] == 0


def test_page_budget_is_bounded(tmp_path: Path) -> None:
    import pytest

    db = populated_db(tmp_path / "news.sqlite3")
    with pytest.raises(ValueError, match="GENERALIZATION_PAGE_LIMIT_INVALID"):
        evaluate_pages(
            db, as_of=date(2024, 2, 15), lookback_days=14,
            pages=9, provider=None,
        )
