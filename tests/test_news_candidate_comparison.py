"""NEWS candidate ranking must be descriptive and never auto-promotable."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_news_research_worker import files, step

from tris_market_event_intelligence.news_candidate_artifact import (
    build_candidate_artifact,
    write_candidate_artifact,
)
from tris_market_event_intelligence.news_candidate_comparison import (
    compare_news_candidates,
    write_comparison_report,
)


def prepared(tmp_path: Path) -> tuple[tuple[Path, Path, Path, Path], list[Path]]:
    inputs = files(tmp_path)
    artifacts = []
    for index in range(2):
        result = step(inputs)
        assert result["status"] == "DESCRIPTIVE_RESEARCH_ONLY"
        artifact_path = tmp_path / f"research-{index}.private.json"
        bundle = build_candidate_artifact(
            cohort_path=inputs[0], plan_path=inputs[1],
            review_path=inputs[2], journal_path=inputs[3],
            experiment_sha256=result["experimentSha256"],
        )
        write_candidate_artifact(artifact_path, bundle)
        artifacts.append(artifact_path)
    return inputs, artifacts


def rank(inputs, candidates):
    return compare_news_candidates(
        candidate_paths=candidates, cohort_path=inputs[0],
        plan_path=inputs[1], review_path=inputs[2], journal_path=inputs[3],
    )


def test_replays_and_compares_without_promotion(tmp_path: Path) -> None:
    inputs, artifacts = prepared(tmp_path)
    originals = [path.read_bytes() for path in (*inputs[:3], *artifacts)]
    result = rank(inputs, artifacts)
    assert result["status"] == "INSUFFICIENT_INDEPENDENT_NEWS_GOLD"
    assert result["candidateCount"] == 2
    assert result["validationArticles"] == 4
    assert result["validationClassCounts"] == {"IMPORTANT": 2, "UNIMPORTANT": 2}
    assert result["promotionAllowed"] is False
    assert result["protectedTestAccessed"] is False
    assert result["modelImprovementValidated"] is False
    assert result["stockPricesUsed"] is False
    assert all(entry["replayVerified"] for entry in result["rankedResearchCandidates"])
    assert result["rankedResearchCandidates"] == sorted(
        result["rankedResearchCandidates"],
        key=lambda x: (x["validationLogLoss"], x["experimentSha256"]),
    )
    assert [path.read_bytes() for path in (*inputs[:3], *artifacts)] == originals


def test_single_or_duplicate_candidate_rejected(tmp_path: Path) -> None:
    inputs, artifacts = prepared(tmp_path)
    with pytest.raises(ValueError, match="2..16"):
        rank(inputs, artifacts[:1])
    with pytest.raises(ValueError, match="distinct"):
        rank(inputs, [artifacts[0], artifacts[0]])


def test_model_changes_fail_closed(tmp_path: Path) -> None:
    inputs, artifacts = prepared(tmp_path)
    doc = json.loads(artifacts[0].read_text())
    doc["payload"]["modelState"]["counts"]["IMPORTANT"]["fictional"] = 123
    artifacts[0].write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="SHA"):
        rank(inputs, artifacts)


def test_new_review_invalidates_prior_rankings(tmp_path: Path) -> None:
    inputs, artifacts = prepared(tmp_path)
    doc = json.loads(inputs[2].read_text())
    doc["judgments"][10]["importance"] = "UNIMPORTANT"
    inputs[2].write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="time-gated"):
        rank(inputs, artifacts)


def test_output_is_create_only(tmp_path: Path) -> None:
    inputs, artifacts = prepared(tmp_path)
    result = rank(inputs, artifacts)
    output = tmp_path / "comparison.private.json"
    write_comparison_report(output, result)
    old = output.read_bytes()
    assert json.loads(old)["promotionAllowed"] is False
    with pytest.raises(ValueError, match="new file"):
        write_comparison_report(output, result)
    assert output.read_bytes() == old


def test_hard_gate_even_with_artificially_strong_metrics(tmp_path: Path) -> None:
    inputs, artifacts = prepared(tmp_path)
    result = rank(inputs, artifacts)
    assert "rankedResearchCandidates" in result
    assert result["humanApprovalRecorded"] is False
    assert result["holdoutNeverUsedForTuningVerified"] is False
    assert result["rankingIsIndependentPerformanceProof"] is False
    assert result["productionModelChanged"] is False
