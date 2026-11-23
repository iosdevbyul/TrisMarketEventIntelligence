"""Regression tests for actual outcome gap attribution and review-only mappings."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tris_market_event_intelligence.market_outcome_gap_review_cli import analyze_gaps


def stored(path: Path, items: list[dict]) -> None:
    path.write_text(json.dumps({
        "schemaVersion": "market_outcome_validation_v1",
        "researchOnly": True,
        "prospectivePredictionAccuracy": None,
        "trainingEligible": False,
        "impactsInspected": len(items),
        "items": items,
    }), encoding="utf-8")


def impact(article: int, target: str, status: str, *, direction: str = "UP") -> dict:
    return {
        "articleId": article, "target": target, "status": status,
        "prediction": direction, "predictionSealedBeforeOutcome": False,
        "evaluationClass": "RETROSPECTIVE_UNSEALED",
        "sourceAuditStatus": "STORED_ORIGINAL_SHA_MATCHED",
    }


def test_gap_counts_and_deduplicated_review_queue(tmp_path: Path) -> None:
    source = tmp_path / "real-shaped-evaluation.json"
    stored(source, [
        impact(1, "Unclear macro target", "UNMAPPED", direction="UP"),
        impact(2, "Unclear macro target", "UNMAPPED", direction="DOWN"),
        impact(3, "Unclear macro target", "UNMAPPED", direction="DOWN"),
        impact(4, "Specific company", "UNMAPPED"),
        impact(5, "Blocked original", "NEWS_SOURCE_NOT_ADMITTED"),
        impact(6, "Priced", "COMPLETED"),
        impact(7, "Late market", "PARTIAL"),
    ])
    original = source.read_bytes()
    result, queue = analyze_gaps(source)
    assert source.read_bytes() == original
    assert result["totalImpacts"] == 7
    assert result["unmappedImpacts"] == 4
    assert result["blockedSourceImpacts"] == 1
    assert result["priceBarGapImpacts"] == 1
    assert result["existingDailyObservedImpacts"] == 1
    assert result["distinctUnmappedTargets"] == 2
    assert result["maxSourceAdmissibleImpactsBeforeAdditionalSourceReview"] == 6
    assert queue[0]["target"] == "Unclear macro target"
    assert queue[0]["impacts"] == 3
    assert queue[0]["articles"] == 3
    assert queue[0]["predictions"] == "DOWN:2;UP:1"
    assert queue[0]["review_decision"] == "PENDING"
    assert queue[0]["reviewed_symbol"] == ""
    assert queue[1]["target"] == "Specific company"
    assert result["mappingApprovedAutomatically"] is False
    assert result["trainingEligible"] is False


@pytest.mark.parametrize("field,value", [
    ("researchOnly", False),
    ("trainingEligible", True),
    ("prospectivePredictionAccuracy", 0.75),
])
def test_reject_prospective_or_training_claims(
    tmp_path: Path, field: str, value: object,
) -> None:
    source = tmp_path / "report.json"
    stored(source, [impact(1, "Target", "UNMAPPED")])
    doc = json.loads(source.read_text())
    doc[field] = value
    source.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="unsealed"):
        analyze_gaps(source)


def test_reject_falsely_sealed_item(tmp_path: Path) -> None:
    source = tmp_path / "report.json"
    item = impact(1, "Target", "UNMAPPED")
    item["predictionSealedBeforeOutcome"] = True
    stored(source, [item])
    with pytest.raises(ValueError, match="prospective"):
        analyze_gaps(source)


def test_reject_incomplete_or_unsupported_results(tmp_path: Path) -> None:
    source = tmp_path / "report.json"
    stored(source, [impact(1, "Target", "UNMAPPED")])
    doc = json.loads(source.read_text())
    doc["impactsInspected"] = 5
    source.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="coverage"):
        analyze_gaps(source)
    stored(source, [impact(1, "Target", "FABRICATED_LABEL")])
    with pytest.raises(ValueError, match="Unexpected"):
        analyze_gaps(source)
