"""Provenance / immutable review queue regression tests for market mappings."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from tris_market_event_intelligence.market_outcome_gap_review_cli import QUEUE_COLUMNS
from tris_market_event_intelligence.reviewed_market_mapping_cli import apply_review
from tris_market_event_intelligence.typed_market_targets_cli import analyze_target_semantics


def _csv(path: Path, columns: list[str], items: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(items)


def fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    source = tmp_path / "outcomes.json"
    source.write_text(json.dumps({
        "schemaVersion": "market_outcome_validation_v1",
        "researchOnly": True,
        "prospectivePredictionAccuracy": None,
        "trainingEligible": False,
        "impactsInspected": 2,
        "items": [],
    }), encoding="utf-8")
    review = tmp_path / "review.csv"
    original = {
        "target": "United Community Bank", "impacts": "2",
        "articles": "2", "article_ids": "1;2", "predictions": "DOWN:1;UP:1",
        "issue": "MAPPING_REQUIRED", "review_decision": "PENDING",
        "reviewed_market": "", "reviewed_symbol": "",
        "reviewed_benchmark": "", "mapping_kind": "",
        "mapping_evidence_url": "", "mapping_verified_at": "",
        "proxy_limitation": "",
    }
    _csv(review, list(QUEUE_COLUMNS), [original])
    gap = tmp_path / "gaps.json"
    gap.write_text(json.dumps({
        "schemaVersion": "market_outcome_gap_review_v1",
        "inputSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "trainingEligible": False,
        "mappingApprovedAutomatically": False,
        "mappingQueue": [{
            key: (int(value) if key in ("impacts", "articles") else value)
            for key, value in original.items()
        }],
    }), encoding="utf-8")
    mappings = tmp_path / "existing-mappings.csv"
    _csv(mappings, ["target", "market", "symbol", "benchmark", "verified_at"], [{
        "target": "Existing issuer", "market": "XNYS", "symbol": "EXIST",
        "benchmark": "SPY", "verified_at": "2026-11-21T00:00:00+00:00",
    }])
    prices = tmp_path / "existing-adjusted.csv"
    _csv(prices, ["symbol", "date", "adjusted_close", "source"], [
        {"symbol": "SPY", "date": "2026-11-20", "adjusted_close": "400",
         "source": "synthetic fixture"},
        {"symbol": "EXIST", "date": "2026-11-20", "adjusted_close": "100",
         "source": "synthetic fixture"},
    ])
    return gap, source, review, mappings, prices


def approve(path: Path, **overrides: str) -> None:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows[0].update({
        "review_decision": "APPROVE",
        "reviewed_market": "XNYS",
        "reviewed_symbol": "UCB",
        "reviewed_benchmark": "SPY",
        "mapping_kind": "DIRECT_EQUITY",
        "mapping_evidence_url": "https://example.org/issuer-listing",
        "mapping_verified_at": "2026-11-23T09:00:00+00:00",
    })
    rows[0].update(overrides)
    _csv(path, list(QUEUE_COLUMNS), rows)



def typed(gap: Path, review: Path) -> Path:
    output = gap.parent / "semantic.json"
    classification = analyze_target_semantics(gap, review)
    output.write_text(json.dumps(classification), encoding="utf-8")
    return output


def test_pending_does_not_approve_or_invent_symbols(tmp_path: Path) -> None:
    args = fixture(tmp_path)
    result, mappings, to_collect = apply_review(*args)
    assert result["newlyApprovedTargets"] == 0
    assert result["pendingTargets"] == ["United Community Bank"]
    assert result["newPriceSymbolsWithoutAnyStoredDailyBars"] == []
    assert to_collect == []
    assert len(mappings) == 1
    assert result["trainingEligible"] is False


def test_approved_target_generates_precise_new_price_collection_plan(
    tmp_path: Path,
) -> None:
    gap, source, review, mappings, prices = fixture(tmp_path)
    source_before = source.read_bytes()
    existing_before = prices.read_bytes()
    approve(review)
    result, new_mapping, collection = apply_review(
        gap, source, review, mappings, prices,
        semantic_plan=typed(gap, review),
    )
    assert source.read_bytes() == source_before
    assert prices.read_bytes() == existing_before
    assert result["newlyApprovedTargets"] == 1
    assert result["approvedNewTargets"] == ["United Community Bank"]
    assert result["newPriceSymbolsWithoutAnyStoredDailyBars"] == ["UCB"]
    assert result["newApprovedTargetsRequiringPriceCollection"] == 1
    assert len(new_mapping) == 2
    assert len(collection) == 1
    assert collection[0]["symbol"] == "UCB"
    assert collection[0]["benchmark"] == "SPY"
    assert collection[0]["evidence_url"].startswith("https://")
    assert result["approvedBySourceDocumentationIndependently"] is False
    assert result["historicalCoverageNotGuaranteed"] is True


def test_approved_existing_price_symbol_skips_unneeded_download(
    tmp_path: Path,
) -> None:
    gap, source, review, mappings, prices = fixture(tmp_path)
    approve(review)
    with prices.open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(["UCB", "2026-11-20", "99", "synthetic fixture"])
    result, mapped, collect = apply_review(
        gap, source, review, mappings, prices, semantic_plan=typed(gap, review),
    )
    assert len(mapped) == 2
    assert collect == []
    assert result["newPriceSymbolsWithoutAnyStoredDailyBars"] == []


@pytest.mark.parametrize("override", [
    {"mapping_evidence_url": ""},
    {"mapping_evidence_url": "http://example.com/a"},
    {"mapping_verified_at": "2026-11-23"},
    {"reviewed_market": "UNKNOWN"},
    {"reviewed_symbol": "SPY"},
    {"mapping_kind": "SECTOR_ETF_PROXY", "proxy_limitation": ""},
])
def test_approval_must_have_documentation_and_explicit_instrument(
    tmp_path: Path, override: dict[str, str],
) -> None:
    gap, source, review, mappings, prices = fixture(tmp_path)
    approve(review, **override)
    with pytest.raises(ValueError):
        apply_review(
            gap, source, review, mappings, prices, semantic_plan=typed(gap, review),
        )


def test_cannot_edit_original_article_or_target_evidence(tmp_path: Path) -> None:
    gap, source, review, mappings, prices = fixture(tmp_path)
    approve(review, impacts="99")
    with pytest.raises(ValueError, match="Original review evidence"):
        apply_review(gap, source, review, mappings, prices)


def test_forged_source_outcomes_are_rejected(tmp_path: Path) -> None:
    gap, source, review, mappings, prices = fixture(tmp_path)
    source.write_text(source.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable source outcome"):
        apply_review(gap, source, review, mappings, prices)


def test_unapproved_mapping_fields_cannot_be_silently_applied(
    tmp_path: Path,
) -> None:
    gap, source, review, mappings, prices = fixture(tmp_path)
    with review.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows[0]["reviewed_symbol"] = "FAKE"
    _csv(review, list(QUEUE_COLUMNS), rows)
    with pytest.raises(ValueError, match="Unapproved"):
        apply_review(gap, source, review, mappings, prices)


def test_not_representable_is_retained_without_made_up_etf(
    tmp_path: Path,
) -> None:
    gap, source, review, mappings, prices = fixture(tmp_path)
    with review.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows[0]["review_decision"] = "NOT_REPRESENTABLE"
    _csv(review, list(QUEUE_COLUMNS), rows)
    result, mapped, collect = apply_review(gap, source, review, mappings, prices)
    assert result["rejectedAsUnrepresentableTargets"] == [
        "United Community Bank"
    ]
    assert len(mapped) == 1
    assert collect == []


def test_unlisted_macro_target_must_never_approve_equity_proxy(tmp_path: Path) -> None:
    gap, source, review, mappings, prices = fixture(tmp_path)
    with review.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows[0]["target"] = "Inflation"
    _csv(review, list(QUEUE_COLUMNS), rows)
    doc = json.loads(gap.read_text())
    doc["mappingQueue"][0]["target"] = "Inflation"
    gap.write_text(json.dumps(doc), encoding="utf-8")
    approve(review)
    with pytest.raises(ValueError, match="not suitable"):
        apply_review(
            gap, source, review, mappings, prices, semantic_plan=typed(gap, review),
        )


def test_arbitrary_approved_stock_needs_semantic_plan(tmp_path: Path) -> None:
    gap, source, review, mappings, prices = fixture(tmp_path)
    approve(review)
    with pytest.raises(ValueError, match="Semantic plan required"):
        apply_review(gap, source, review, mappings, prices)


def test_forged_semantic_parent_ticker_cannot_approve_stock(tmp_path: Path) -> None:
    gap, source, review, mappings, prices = fixture(tmp_path)
    approve(review)
    plan = typed(gap, review)
    metadata = json.loads(plan.read_text(encoding="utf-8"))
    metadata["targets"][0]["candidateRelatedListedParentSymbol"] = "FAKE"
    plan.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="source catalog"):
        apply_review(
            gap, source, review, mappings, prices, semantic_plan=plan,
        )
