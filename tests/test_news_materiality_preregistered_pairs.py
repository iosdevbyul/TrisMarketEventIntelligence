"""Preregistered NEWS pairs are chosen without using model scores or TEST."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_news_materiality_transition_review import _fixture

from tris_market_event_intelligence.news_materiality_preregistered_pairs import (
    evaluate_preregistered_pairs,
    prepare_preregistered_pairs,
)


def _prepare(paths):
    return prepare_preregistered_pairs(cohort_path=paths[0], holdout_path=paths[1])


def _audit(paths, human, report):
    return evaluate_preregistered_pairs(
        cohort_path=paths[0], holdout_path=paths[1],
        review_path=human, report_path=report,
    )


def test_pregeneration_is_model_independent_blind_and_repeatable(tmp_path: Path):
    paths, report = _fixture(tmp_path)
    original_files = [p.read_bytes() for p in paths[:3]]
    first = _prepare(paths)
    assert first == _prepare(paths)
    assert 1 <= len(first["judgments"]) <= 3
    assert first["candidateSelectionIndependentOfModel"] is True
    assert first["modelMagnitudesExposedInTemplate"] is False
    assert "potentialMagnitude" not in json.dumps(first)
    assert "ordinalDifference" not in json.dumps(first)
    assert all(7 <= j["earlierArticleId"] <= 10 for j in first["judgments"])
    assert all(7 <= j["laterArticleId"] <= 10 for j in first["judgments"])
    assert report.is_file()
    assert [p.read_bytes() for p in paths[:3]] == original_files


def test_model_predictions_cannot_change_preselected_pairs(tmp_path: Path):
    paths, report = _fixture(tmp_path)
    first = _prepare(paths)
    payload = json.loads(report.read_text())
    for hypothesis in payload["result"]["hypotheses"]:
        hypothesis["potential_magnitude"] = "UNASSESSED"
        hypothesis["magnitude_basis"] = "NOT_ASSESSED"
        hypothesis["direction"] = "UNCERTAIN"
    # No model report is ever opened to select pairs; even nonsensical
    # output values cannot affect deterministic source-only selection.
    report.write_text(json.dumps(payload))
    assert _prepare(paths) == first


def test_human_review_only_scores_same_event_pairs(tmp_path: Path):
    paths, report = _fixture(tmp_path)
    pkt = _prepare(paths)
    for j in pkt["judgments"]:
        j.update({
            "relation": "SAME_EVENT", "humanChange": "INCREASE",
            "reviewer": "fixture-editor",
            "reviewedAt": "2026-11-13T00:00:00+00:00",
            "reason": "Editorial judgment on fictional related news and potential impact.",
        })
    human = tmp_path / "human-pairs.json"
    human.write_text(json.dumps(pkt))
    out = _audit(paths, human, report)
    assert out["scoredSameEventPairs"] == len(pkt["judgments"])
    assert 0 <= out["sameEventDirectionAgreement"] <= 1
    assert out["status"] == "INSUFFICIENT_HUMAN_PAIR_GOLD"
    assert out["testGoldScored"] is False
    assert out["promotionAllowed"] is False
    assert out["modelPerformanceImprovementValidated"] is False
    assert not paths[3].exists()


def test_different_or_uncertain_event_not_scored(tmp_path: Path):
    paths, report = _fixture(tmp_path)
    pkt = _prepare(paths)
    pkt["judgments"][0].update({
        "relation": "DIFFERENT_EVENT", "humanChange": "UNASSESSABLE",
        "reviewer": "fixture-editor",
        "reviewedAt": "2026-11-13T00:00:00+00:00",
        "reason": "The two reports do not describe one continuing event.",
    })
    human = tmp_path / "human-pairs.json"
    human.write_text(json.dumps(pkt))
    out = _audit(paths, human, report)
    assert out["differentEvents"] == 1
    assert out["scoredSameEventPairs"] == 0
    assert out["unreviewedPairs"] == len(pkt["judgments"]) - 1
    pkt["judgments"][0]["humanChange"] = "INCREASE"
    human.write_text(json.dumps(pkt))
    with pytest.raises(ValueError, match="Unrelated or uncertain"):
        _audit(paths, human, report)


def test_review_tampering_and_test_partition_fail_closed(tmp_path: Path):
    paths, report = _fixture(tmp_path)
    pkt = _prepare(paths)
    human = tmp_path / "human-pairs.json"
    pkt["judgments"][0]["earlierArticleId"] = 11
    human.write_text(json.dumps(pkt))
    with pytest.raises(ValueError, match="identities"):
        _audit(paths, human, report)
    pkt = _prepare(paths)
    pkt["cohortSha256"] = "0" * 64
    human.write_text(json.dumps(pkt))
    with pytest.raises(ValueError, match="not pinned"):
        _audit(paths, human, report)
    pkt = _prepare(paths)
    plan = json.loads(paths[1].read_text())
    for row in plan["items"]:
        if row["articleId"] == 11:
            row["split"] = "VALIDATION"
    # Do not write inconsistent counts to bypass the holdout checker.
    plan["counts"] = {"TRAIN": 6, "VALIDATION": 5, "TEST": 1}
    paths[1].write_text(json.dumps(plan))
    human.write_text(json.dumps(pkt))
    with pytest.raises(ValueError, match="not pinned"):
        _audit(paths, human, report)


def test_source_mutation_refused(tmp_path: Path):
    paths, _ = _fixture(tmp_path)
    doc = json.loads(paths[0].read_text())
    doc["items"][7]["title"] = "Hacked headline"
    paths[0].write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="hash mismatch"):
        _prepare(paths)
