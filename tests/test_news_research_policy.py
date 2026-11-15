"""Finite exploratory hypothesis budget must not leak VALIDATION/TEST scoring."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_news_research_worker import TRAIN_END, VALIDATION_END, files

from tris_market_event_intelligence.news_holdout_plan import plan_news_holdout
from tris_market_event_intelligence.news_research_policy import (
    MAX_HYPOTHESES,
    _representation,
    audit_policy,
    run_policy_once,
)


def attempt(paths):
    return run_policy_once(
        cohort_path=paths[0], plan_path=paths[1],
        review_path=paths[2], journal_path=paths[3],
    )


def test_hypothesis_policy_exhausts_and_resume_is_idempotent(tmp_path: Path):
    paths = files(tmp_path)
    original = [path.read_bytes() for path in paths[:3]]
    hashes = set()
    for _ in range(MAX_HYPOTHESES):
        outcome = attempt(paths)
        assert outcome["status"] == "TRAIN_ONLY_HYPOTHESIS_RESEARCH"
        assert outcome["validationGoldScored"] is False
        assert outcome["testGoldScored"] is False
        assert outcome["promotionAllowed"] is False
        assert outcome["independentImprovementVerified"] is False
        assert outcome["experimentSha256"] not in hashes
        hashes.add(outcome["experimentSha256"])
    assert len(hashes) == 12
    exhausted = attempt(paths)
    assert exhausted["status"] == "POLICY_HYPOTHESES_EXHAUSTED"
    assert exhausted["newResearchExperiment"] is False
    audit = audit_policy(paths[3])
    assert audit["totalExperiments"] == 12
    assert audit["promotionAllowed"] is False
    assert len(audit["researchOnlyRanking"]) == 12
    assert [p.read_bytes() for p in paths[:3]] == original


def test_feature_representations_differ_and_are_deterministic():
    words = ["economic", "economic", "inflation"]
    assert _representation(words, "WORD_PRESENCE", 32) == [
        "economic", "inflation",
    ]
    gram = _representation(words, "CHAR_TRIGRAM_PRESENCE", 32)
    assert gram[0] == "#eco"
    assert len(gram) == len(set(gram))
    assert gram != words
    with pytest.raises(ValueError):
        _representation(words, "INVALID", 32)


def test_validation_relabel_cannot_change_train_fold_score(tmp_path: Path):
    paths = files(tmp_path)
    initial = attempt(paths)
    review = json.loads(paths[2].read_text())
    for item in review["judgments"][6:]:
        if item["importance"] in ("IMPORTANT", "UNIMPORTANT"):
            item["importance"] = (
                "UNIMPORTANT" if item["importance"] == "IMPORTANT"
                else "IMPORTANT"
            )
    paths[2].write_text(json.dumps(review))
    plan = plan_news_holdout(
        paths[0], train_end=TRAIN_END, validation_end=VALIDATION_END,
        review_path=paths[2],
    )
    paths[1].write_text(json.dumps(plan))
    changed = attempt(paths)
    assert changed["featureMode"] == initial["featureMode"]
    assert changed["trainFoldLogLoss"] == initial["trainFoldLogLoss"]
    assert changed["validationGoldScored"] is False


def test_invalid_holdout_rejected_before_journal(tmp_path: Path):
    paths = files(tmp_path)
    plan = json.loads(paths[1].read_text())
    plan["humanReviewTimeGateApplied"] = False
    paths[1].write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="time-gated"):
        attempt(paths)
    assert not paths[3].exists()
