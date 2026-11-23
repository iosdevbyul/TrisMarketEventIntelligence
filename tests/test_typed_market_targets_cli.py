"""Actual-target semantic safety and source-provenance regression fixtures."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from tris_market_event_intelligence.market_outcome_gap_review_cli import QUEUE_COLUMNS
from tris_market_event_intelligence.typed_market_targets_cli import (
    KNOWN,
    analyze_target_semantics,
)


def fixture(tmp_path: Path, targets: list[str]) -> tuple[Path, Path]:
    gap = tmp_path / "gaps.json"
    queue = tmp_path / "queue.csv"
    entries = []
    for index, target in enumerate(targets, 1):
        entries.append({
            "target": target, "impacts": 1, "articles": 1,
            "article_ids": str(index), "predictions": "UP:1",
            "issue": "MAPPING_REQUIRED", "review_decision": "PENDING",
            "reviewed_market": "", "reviewed_symbol": "",
            "reviewed_benchmark": "", "mapping_kind": "",
            "mapping_evidence_url": "", "mapping_verified_at": "",
            "proxy_limitation": "",
        })
    with queue.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUEUE_COLUMNS)
        writer.writeheader()
        writer.writerows(entries)
    gap.write_text(json.dumps({
        "schemaVersion": "market_outcome_gap_review_v1",
        "mappingApprovedAutomatically": False,
        "trainingEligible": False,
        "unmappedImpacts": len(entries),
        "mappingQueue": entries,
    }), encoding="utf-8")
    return gap, queue


def test_all_25_exact_actual_targets_are_classified_without_stock_fabrication(
    tmp_path: Path,
) -> None:
    gap, queue = fixture(tmp_path, list(KNOWN))
    report = analyze_target_semantics(gap, queue)
    assert report["targetCount"] == 25
    assert report["impactCount"] == 25
    assert report["anyStockMappingApproved"] is False
    assert report["newMarketPriceObservations"] == 0
    assert report["trainingEligible"] is False
    assert all(item["legacyStockReturnEligible"] is False for item in report["targets"])
    categories = {item["target"]: item for item in report["targets"]}
    assert categories["Inflation"]["targetKind"] == "MACRO_RELEASE"
    assert categories["Treasury Yields"]["targetKind"] == "YIELD_SERIES"
    assert categories["Equity Markets (South Korea, Taiwan)"]["targetKind"] == "MULTI_MARKET"
    assert categories["MapleMark"]["targetKind"] == "ACQUISITION_TARGET_BANK"
    assert categories["United Community Bank"]["candidateRelatedListedParentSymbol"] == "UCB"
    assert categories["United Community Bank"]["securityMappingApproved"] is False
    assert report["queueSha256"] == hashlib.sha256(queue.read_bytes()).hexdigest()


def test_unknown_target_cannot_be_automatically_mapped(tmp_path: Path) -> None:
    gap, queue = fixture(tmp_path, ["Brand-new obscure asset"])
    result = analyze_target_semantics(gap, queue)
    assert result["targets"][0]["targetKind"] == "UNCLASSIFIED"
    assert result["targets"][0]["candidateRelatedListedParentSymbol"] is None
    assert result["targets"][0]["legacyStockReturnEligible"] is False


def test_tampered_source_target_or_counts_fails_closed(tmp_path: Path) -> None:
    gap, queue = fixture(tmp_path, ["Financial Markets"])
    with queue.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["article_ids"] = "999"
    with queue.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUEUE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="changed"):
        analyze_target_semantics(gap, queue)


def test_missing_or_extra_queue_target_is_rejected(tmp_path: Path) -> None:
    gap, queue = fixture(tmp_path, ["Financial Markets"])
    with queue.open("a", encoding="utf-8") as handle:
        handle.write("bad,not-valid\n")
    with pytest.raises(ValueError):
        analyze_target_semantics(gap, queue)
