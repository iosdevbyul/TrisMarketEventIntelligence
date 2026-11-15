"""TRAIN-only news exploration must never tune against VALIDATION or TEST gold."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_news_research_worker import TRAIN_END, VALIDATION_END, files, step

from tris_market_event_intelligence.news_holdout_plan import plan_news_holdout
from tris_market_event_intelligence.news_train_cv_exploration import (
    FEATURES,
    TOKEN_LIMITS,
    _features,
    _folds,
    audit_train_exploration,
    explore_news_train_once,
)


def explore(inputs):
    return explore_news_train_once(
        cohort_path=inputs[0], plan_path=inputs[1],
        review_path=inputs[2], journal_path=inputs[3],
    )


def test_research_on_unchanged_history_deduplicates_18_configurations(
    tmp_path: Path,
) -> None:
    paths = files(tmp_path)
    originals = [path.read_bytes() for path in paths[:3]]
    seen = set()
    for _ in range(18):
        attempt = explore(paths)
        assert attempt["status"] == "TRAIN_ONLY_CV_EXPLORATORY"
        assert attempt["trainOnlyFoldCount"] == 3
        assert attempt["trainingArticles"] == 6
        assert attempt["validationGoldScored"] is False
        assert attempt["testGoldScored"] is False
        assert attempt["modelImprovementValidated"] is False
        assert attempt["stockPricesUsed"] is False
        assert attempt["experimentSha256"] not in seen
        seen.add(attempt["experimentSha256"])
    assert len(seen) == len(FEATURES) * len(TOKEN_LIMITS) * 3
    exhausted = explore(paths)
    assert exhausted["status"] == "TRAIN_CV_SEARCH_SPACE_EXHAUSTED"
    assert exhausted["newResearchExperiment"] is False
    audit = audit_train_exploration(paths[3])
    assert audit["totalExperiments"] == 18
    assert len({entry["experimentSha256"] for entry in audit["items"]}) == 18
    assert [path.read_bytes() for path in paths[:3]] == originals
    # Existing legacy five-alpha journal remains a separate experiment family.
    legacy = step(paths)
    assert legacy["status"] == "DESCRIPTIVE_RESEARCH_ONLY"
    assert audit_train_exploration(paths[3])["totalExperiments"] == 18


def test_train_cv_does_not_change_when_validation_or_test_gold_changes(
    tmp_path: Path,
) -> None:
    paths = files(tmp_path)
    first = explore(paths)
    review = json.loads(paths[2].read_text())
    for row in review["judgments"][6:]:
        row["importance"] = (
            "UNIMPORTANT" if row["importance"] == "IMPORTANT" else "IMPORTANT"
        )
    paths[2].write_text(json.dumps(review))
    plan = plan_news_holdout(
        paths[0], train_end=TRAIN_END,
        validation_end=VALIDATION_END, review_path=paths[2],
    )
    paths[1].write_text(json.dumps(plan))
    second = explore(paths)
    assert second["trainOnlyCrossValidationLogLoss"] == (
        first["trainOnlyCrossValidationLogLoss"]
    )
    assert second["featureMode"] == first["featureMode"]
    assert second["tokenLimit"] == first["tokenLimit"]
    assert second["validationGoldScored"] is False
    assert second["testGoldScored"] is False


def test_train_gold_missing_blocks_without_fake_experiment(tmp_path: Path) -> None:
    paths = files(tmp_path)
    review = json.loads(paths[2].read_text())
    for row in review["judgments"][:5]:
        for field in ("importance", "sourceAlignment", "reviewer", "reviewedAt", "reason"):
            row[field] = None
    paths[2].write_text(json.dumps(review))
    plan = plan_news_holdout(
        paths[0], train_end=TRAIN_END,
        validation_end=VALIDATION_END, review_path=paths[2],
    )
    paths[1].write_text(json.dumps(plan))
    response = explore(paths)
    assert response["status"] == "BLOCKED_INSUFFICIENT_TRAIN_HUMAN_GOLD"
    assert response["modelTrained"] is False
    assert not paths[3].exists()


def test_tampered_holdout_provenance_fails_before_writes(tmp_path: Path) -> None:
    paths = files(tmp_path)
    plan = json.loads(paths[1].read_text())
    plan["humanReviewTimeGateApplied"] = False
    paths[1].write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="time-gated"):
        explore(paths)
    assert not paths[3].exists()


def test_stratified_folds_train_only_and_features_are_deterministic() -> None:
    data = [
        (["headline", "news", str(index)], label)
        for label in ("IMPORTANT", "UNIMPORTANT")
        for index in range(4)
    ]
    folds = _folds(data)
    assert len(folds) == 3
    for train, held_out in folds:
        assert set(train).isdisjoint(held_out)
        assert sorted(train + held_out) == list(range(len(data)))
        assert {data[index][1] for index in train} == {"IMPORTANT", "UNIMPORTANT"}
        assert {data[index][1] for index in held_out} == {"IMPORTANT", "UNIMPORTANT"}
    words = ["Economic", "news", "matters", "today"]
    assert _features(words, mode="UNIGRAM", token_limit=32) == words
    assert _features(words, mode="UNIGRAM_BIGRAM", token_limit=32) == [
        *words, "Economic~news", "news~matters", "matters~today",
    ]


def test_disallowed_feature_settings_rejected() -> None:
    with pytest.raises(ValueError, match="Unrecognized"):
        _features(["news"], mode="UNIGRAM", token_limit=10000)
    with pytest.raises(ValueError, match="Unrecognized"):
        _features(["news"], mode="UNVERIFIED", token_limit=32)
