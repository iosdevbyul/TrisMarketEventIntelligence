"""NEWS learning readiness is descriptive and cannot grant deployment rights."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_news_research_worker import TRAIN_END, VALIDATION_END, files

from tris_market_event_intelligence.news_holdout_plan import plan_news_holdout
from tris_market_event_intelligence.news_learning_readiness import (
    inspect_news_learning_readiness,
)


def readiness(paths, **kwargs):
    return inspect_news_learning_readiness(
        cohort_path=paths[0], plan_path=paths[1], review_path=paths[2],
        **kwargs,
    )


def test_preflight_checks_pinned_news_without_writing(tmp_path: Path) -> None:
    paths = files(tmp_path)
    original = [p.read_bytes() for p in paths[:3]]
    result = readiness(paths)
    assert result["schemaVersion"] == "news_learning_readiness_v1"
    assert result["trainingArticlesWithTimeGatedHumanLabels"] == 6
    assert result["trainFoldResearchFeasible"] is True
    assert result["validationGoldScored"] is False
    assert result["testGoldScored"] is False
    assert result["humanGoldIndependentlyAdjudicated"] is False
    assert result["noLexicalCandidatesProvesNoLeakage"] is False
    assert result["promotionAllowed"] is False
    assert not paths[3].exists()
    assert [p.read_bytes() for p in paths[:3]] == original


def test_insufficient_train_human_labels_block(tmp_path: Path) -> None:
    paths = files(tmp_path)
    review = json.loads(paths[2].read_text())
    for row in review["judgments"][:5]:
        for field in (
            "importance", "sourceAlignment", "reviewer", "reviewedAt", "reason",
        ):
            row[field] = None
    paths[2].write_text(json.dumps(review))
    plan = plan_news_holdout(
        paths[0], train_end=TRAIN_END, validation_end=VALIDATION_END,
        review_path=paths[2],
    )
    paths[1].write_text(json.dumps(plan))
    result = readiness(paths)
    assert result["trainFoldResearchFeasible"] is False
    assert "INSUFFICIENT_TRAIN_HUMAN_GOLD" in result["blockingFindings"]
    assert result["promotionAllowed"] is False


def test_cross_partition_lexical_review_blocks(tmp_path: Path, monkeypatch) -> None:
    paths = files(tmp_path)
    def flagged(**_kwargs):
        return {
            "crossPartitionCandidates": 2,
            "status": "REVIEW_REQUIRED",
            "threshold": 0.78,
        }

    monkeypatch.setattr(
        "tris_market_event_intelligence.news_learning_readiness."
        "audit_news_near_duplicates", flagged,
    )
    result = readiness(paths)
    assert result["crossSplitLexicalReviewCandidates"] == 2
    assert "CROSS_SPLIT_LEXICAL_REVIEW_REQUIRED" in result["blockingFindings"]
    assert result["status"] == "BLOCKED_REVIEW_OR_RESEARCH_PREREQUISITES"


def test_candidate_inputs_must_be_complete(tmp_path: Path) -> None:
    paths = files(tmp_path)
    with pytest.raises(ValueError, match="together"):
        readiness(paths, lock_path=tmp_path / "lock.json")
    with pytest.raises(ValueError, match="requires"):
        readiness(paths, usage_ledger_path=tmp_path / "ledger.sqlite3")
    assert not paths[3].exists()


def test_tampered_holdout_is_refused(tmp_path: Path) -> None:
    paths = files(tmp_path)
    plan = json.loads(paths[1].read_text())
    plan["humanReviewTimeGateApplied"] = False
    paths[1].write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="time-gated"):
        readiness(paths)
    assert not paths[3].exists()
